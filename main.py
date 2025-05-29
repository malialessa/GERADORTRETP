"""
Projeto: Gerador de ETP e Termo de Referência com IA para Documentos Xertica.ai
Módulo: main.py
Descrição: Backend da aplicação FastAPI que orquestra a geração de documentos ETP (Estudo Técnico Preliminar)
           e TR (Termo de Referência) utilizando o modelo Gemini do Google Vertex AI.
           Integra-se com Google Cloud Storage para armazenamento de anexos e Google Docs API para
           criação e formatação dos documentos finais.
Autor: Xertica.ai - Assistente de IA
Data: 2024-05-23
Versão: 0.2.0 (Com carregamento expandido de GCS)
"""

import os
import logging
import json
import io
from datetime import date
import re
import sys # Adicionado para sys.exit em caso de falha crítica na inicialização

from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Union # Union adicionado para tipagem
from dotenv import load_dotenv

# Google Cloud Imports
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pypdf import PdfReader

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente (para desenvolvimento local)
load_dotenv()

app = FastAPI(
    title="Gerador de ETP e TR Xertica.ai",
    description="Backend inteligente para gerar documentos ETP e TR com IA da Xertica.ai.",
    version="0.2.0"
)

# Configurações CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variáveis de Ambiente e Inicialização de Clientes GCP / Vertex AI ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.getenv("GCP_PROJECT_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "docsorgaospublicos")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-001")

if not GCP_PROJECT_ID:
    logger.critical("GCP_PROJECT_ID não está configurado. A aplicação não pode iniciar.")
    sys.exit("FATAL: GCP_PROJECT_ID não configurado.")
if not GCP_PROJECT_LOCATION:
    logger.critical("GCP_PROJECT_LOCATION não está configurado. A aplicação não pode iniciar.")
    sys.exit("FATAL: GCP_PROJECT_LOCATION não configurado.")
if not GCS_BUCKET_NAME:
    logger.warning("GCS_BUCKET_NAME não está configurado. Funcionalidades de GCS podem falhar.")

gemini_model = None
_generation_config = None
storage_client = None

try:
    logger.info(f"Inicializando Vertex AI com projeto '{GCP_PROJECT_ID}' e localização '{GCP_PROJECT_LOCATION}'.")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_PROJECT_LOCATION)
    logger.info(f"Carregando modelo Gemini: '{GEMINI_MODEL_NAME}'.")
    gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
    _generation_config = GenerationConfig(
        temperature=0.7,
        max_output_tokens=8192,
        response_mime_type="application/json"
    )
    logger.info(f"Modelo Gemini '{GEMINI_MODEL_NAME}' carregado e configurado.")
except Exception as e:
    logger.exception(f"Erro CRÍTICO ao inicializar Vertex AI ou carregar modelo Gemini: {e}")
    sys.exit(f"FATAL: Falha ao iniciar LLM: {e}.")

try:
    logger.info(f"Inicializando cliente Google Cloud Storage para o projeto '{GCP_PROJECT_ID}'.")
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Cliente Google Cloud Storage inicializado com sucesso.")
except Exception as e:
    logger.exception(f"Erro CRÍTICO ao inicializar cliente Google Cloud Storage: {e}")
    sys.exit(f"FATAL: Falha ao inicializar GCS: {e}.")

def authenticate_google_docs_and_drive() -> tuple[Optional[object], Optional[object]]:
    try:
        docs_service = build('docs', 'v1', cache_discovery=False)
        drive_service = build('drive', 'v3', cache_discovery=False)
        logger.info("Serviços Google Docs e Drive API inicializados com sucesso.")
        return docs_service, drive_service
    except Exception as e:
        logger.exception(f"Erro ao autenticar/inicializar Google Docs/Drive APIs: {e}")
        return None, None

def get_gcs_file_content(file_path: str) -> Optional[str]:
    if not storage_client:
        logger.error("GCS client não inicializado. Não é possível ler o arquivo.")
        return None
    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME não configurado. Não é possível ler o arquivo.")
        return None
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
        content = None
        if blob.exists():
            for encoding in encodings_to_try:
                try:
                    content = blob.download_as_text(encoding=encoding)
                    logger.info(f"Conteúdo de GCS://{GCS_BUCKET_NAME}/{file_path} lido com sucesso ({len(content)} chars) usando encoding {encoding}.")
                    return content
                except UnicodeDecodeError:
                    logger.warning(f"Falha ao decodificar GCS://{GCS_BUCKET_NAME}/{file_path} com {encoding}.")
                except Exception as e_enc:
                    logger.exception(f"Erro inesperado ao tentar ler GCS://{GCS_BUCKET_NAME}/{file_path} com encoding {encoding}: {e_enc}")
                    break
            if content is None:
                 logger.error(f"Não foi possível decodificar o arquivo GCS://{GCS_BUCKET_NAME}/{file_path} com os encodings testados.")
                 return f"ERRO_DECODIFICACAO: Não foi possível ler o conteúdo do arquivo {file_path} devido a problemas de encoding."
        else:
            logger.warning(f"Arquivo não encontrado no GCS: gs://{GCS_BUCKET_NAME}/{file_path}")
            return None
    except Exception as e:
        logger.exception(f"Erro crítico ao ler arquivo GCS gs://{GCS_BUCKET_NAME}/{file_path}: {e}")
        return None

async def upload_file_to_gcs(upload_file: UploadFile, destination_path: str) -> Optional[str]:
    if not storage_client:
        logger.error("GCS client não inicializado. Upload falhou.")
        return None
    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME não configurado. Upload falhou.")
        return None
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_path)
        contents = await upload_file.read()
        await upload_file.seek(0)
        blob.upload_from_string(contents, content_type=upload_file.content_type)
        logger.info(f"Arquivo '{upload_file.filename}' carregado para GCS://{GCS_BUCKET_NAME}/{destination_path}.")
        return f"gs://{GCS_BUCKET_NAME}/{destination_path}"
    except Exception as e:
        logger.exception(f"Erro ao fazer upload do arquivo '{upload_file.filename}' para GCS: {e}")
        return None

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    logger.info(f"Iniciando extração de texto do PDF: {pdf_file.filename}")
    try:
        contents = await pdf_file.read()
        await pdf_file.seek(0)
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page_num, page in enumerate(reader.pages):
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + "\n"
            else:
                logger.warning(f"Nenhum texto extraído da página {page_num + 1} do PDF {pdf_file.filename}.")
        logger.info(f"Texto extraído do PDF {pdf_file.filename} (tamanho total: {len(text)} caracteres)")
        if not text.strip():
            logger.warning(f"O texto extraído de {pdf_file.filename} está vazio ou contém apenas espaços em branco.")
            return (f"**AVISO_PDF_VAZIO:** Não foi possível extrair texto legível do PDF '{pdf_file.filename}'. "
                    f"O arquivo pode ser um PDF de imagem ou ter um formato que dificulta a extração. "
                    f"O conteúdo deste arquivo não estará disponível para análise detalhada.")
        return text
    except Exception as e:
        logger.exception(f"Erro crítico ao extrair texto do PDF {pdf_file.filename}: {e}")
        return (f"**ERRO_EXTRACAO_PDF:** Ocorreu um erro ao processar o PDF '{pdf_file.filename}': {str(e)}. "
                f"O conteúdo deste PDF não pôde ser analisado.")

def apply_basic_markdown_to_docs_requests(markdown_content: str) -> List[Dict]:
    requests: List[Dict[str, Union[str, Dict]]] = []
    lines = markdown_content.split('\n')
    current_index = 1
    for line in lines:
        line_stripped = line.strip()
        if line_stripped == "<NEWPAGE>":
            requests.append({"insertPageBreak": {"location": {"index": current_index -1 if current_index > 1 else 1 }}})
            continue
        text_to_insert = line_stripped + "\n"
        requests.append({"insertText": {"location": {"index": current_index}, "text": text_to_insert}})
        start_text_index = current_index
        end_text_index = start_text_index + len(line_stripped)
        if line_stripped.startswith('### '):
            requests.append({"updateParagraphStyle": {"range": {"startIndex": start_text_index, "endIndex": end_text_index},"paragraphStyle": {"namedStyleType": "HEADING_3"},"fields": "namedStyleType"}})
        elif line_stripped.startswith('## '):
            requests.append({"updateParagraphStyle": {"range": {"startIndex": start_text_index, "endIndex": end_text_index},"paragraphStyle": {"namedStyleType": "HEADING_2"},"fields": "namedStyleType"}})
        elif line_stripped.startswith('# '):
            requests.append({"updateParagraphStyle": {"range": {"startIndex": start_text_index, "endIndex": end_text_index},"paragraphStyle": {"namedStyleType": "HEADING_1"},"fields": "namedStyleType"}})
        elif line_stripped.startswith('* ') or line_stripped.startswith('- '):
            requests.append({"createParagraphBullets": {"range": {"startIndex": start_text_index, "endIndex": start_text_index + len(text_to_insert)},"bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"}})
        
        text_content_for_bold = line_stripped
        offset = 0
        if line_stripped.startswith('### '): text_content_for_bold = line_stripped[4:]; offset = 4
        elif line_stripped.startswith('## '): text_content_for_bold = line_stripped[3:]; offset = 3
        elif line_stripped.startswith('# '): text_content_for_bold = line_stripped[2:]; offset = 2
        elif line_stripped.startswith('* ') or line_stripped.startswith('- '): text_content_for_bold = line_stripped[2:]; offset = 2
        for match in re.finditer(r'\*\*(.*?)\*\*', text_content_for_bold):
            bold_start_index_in_line = match.start(1) - 2
            bold_end_index_in_line = match.end(1)
            actual_bold_start = start_text_index + offset + bold_start_index_in_line
            actual_bold_end = start_text_index + offset + bold_end_index_in_line
            if actual_bold_start < actual_bold_end :
                requests.append({"updateTextStyle": {"range": {"startIndex": actual_bold_start, "endIndex": actual_bold_end},"textStyle": {"bold": True},"fields": "bold"}})
        current_index += len(text_to_insert)
    return requests

async def generate_etp_tr_content_with_gemini(llm_context_data: Dict) -> Dict:
    if not gemini_model:
        logger.error("Modelo Gemini não inicializado. Não é possível gerar conteúdo.")
        raise HTTPException(status_code=503, detail="Serviço de IA (LLM) não configurado ou falhou ao iniciar.")

    logger.info("Iniciando preparação do prompt e chamada ao Gemini para geração de ETP/TR.")
    
    # =======================================================================
    # PASSO 1: MONTAGEM DE TODAS AS VARIÁVEIS DE CONTEXTO PARA O PROMPT
    # Coloque aqui TODA a sua lógica para definir:
    # gcs_accel_str, gcs_legal_str, orgao_nome, titulo_projeto, 
    # justificativa_necessidade, objetivo_geral, prazos_estimados, 
    # valor_estimado_input, modelo_licitacao, parcelamento_contratacao, 
    # justificativa_parcelamento, contexto_geral_orgao, today, mes_extenso, 
    # ano_atual, esfera_administrativa, local_etp_full_placeholder, 
    # accelerator_details_prompt_section, proposta_comercial_content, 
    # proposta_tecnica_content, price_map_to_use_template, 
    # produtos_originais_display_str, abes_certs_str, coe_content_str
    # Exemplo de como começar:
    # =======================================================================
    gcs_accel_str_parts = []
    for product_key, content in llm_context_data.get('gcs_accelerator_content', {}).items():
        if content:
            gcs_accel_str_parts.append(f"Conteúdo GCS - Acelerador {product_key}:\n{content}\n---\n")
    gcs_accel_str = "\n".join(gcs_accel_str_parts) if gcs_accel_str_parts else "Nenhum conteúdo de acelerador do GCS fornecido.\n"

    gcs_legal_str_parts = []
    sorted_legal_items = sorted(llm_context_data.get('gcs_legal_context_content', {}).items())
    for file_name, content in sorted_legal_items:
        if content:
            gcs_legal_str_parts.append(f"Conteúdo GCS - Documento Legal/Contexto Adicional ({file_name}):\n{content}\n---\n")
    gcs_legal_str = "\n".join(gcs_legal_str_parts) if gcs_legal_str_parts else "Nenhum conteúdo legal/contextual do GCS fornecido.\n"
    
    orgao_nome = llm_context_data.get('orgaoSolicitante', 'o Órgão Solicitante')
    titulo_projeto = llm_context_data.get('tituloProjeto', 'uma iniciativa')
    justificativa_necessidade = llm_context_data.get('justificativaNecessidade', 'um problema genérico.')
    objetivo_geral = llm_context_data.get('objetivoGeral', 'um objetivo ambicioso.')
    prazos_estimados = llm_context_data.get('prazosEstimados', 'prazos a serem definidos.')
    valor_estimado_input = llm_context_data.get('valorEstimado')
    modelo_licitacao = llm_context_data.get('modeloLicitacao', 'uma modalidade padrão.')
    parcelamento_contratacao = llm_context_data.get('parcelamentoContratacao', 'Não especificado.')
    justificativa_parcelamento = llm_context_data.get('justificativaParcelamento', 'Não se aplica.')
    contexto_geral_orgao = llm_context_data.get('contextoGeralOrgao', '')
    today = date.today()
    meses_pt = {1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"}
    mes_extenso = meses_pt[today.month]
    ano_atual = today.year
    esfera_administrativa = "Federal"
    orgao_nome_lower = orgao_nome.lower()
    if any(term in orgao_nome_lower for term in ["municipal", "pref.", "prefeitura"]):
        esfera_administrativa = "Municipal"
    elif any(term in orgao_nome_lower for term in ["estadual", "governo do estado", "secretaria de estado", "tj", "tribunal de justiça", "estado de"]):
        esfera_administrativa = "Estadual"
    local_etp_full_placeholder = f"[LOCAL PADRÃO - CIDADE/UF], {today.day} de {mes_extenso} de {ano_atual}" # Você pode querer refinar isso
    
    accelerator_details_prompt_list = []
    produtos_selecionados_normalizados = llm_context_data.get("produtosXertica", [])
    for product_name_normalized in produtos_selecionados_normalizados:
        product_name_original = product_name_normalized.replace('_', ' ')
        integration_key = f"integracao_{product_name_normalized}"
        user_integration_detail = llm_context_data.get(integration_key, "").strip()
        bc_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name_original} (BC)", "Dados do Battle Card não disponíveis.")
        ds_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name_original} (DS)", "Dados do Data Sheet não disponíveis.")
        op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name_original} (OP)",
                                 llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name_original} (OP_GCP)",
                                 llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name_original} (OP_GMP)",
                                 llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name_original} (OP_GWS)",
                                 "Dados do Plano Operacional não disponíveis."))))
        bc_summary = (bc_content_prod_raw[:min(800, len(bc_content_prod_raw))] + "...") if len(bc_content_prod_raw) > 800 else bc_content_prod_raw
        ds_summary = (ds_content_prod_raw[:min(800, len(ds_content_prod_raw))] + "...") if len(ds_content_prod_raw) > 800 else ds_content_prod_raw
        op_summary = (op_content_prod_raw[:min(800, len(op_content_prod_raw))] + "...") if len(op_content_prod_raw) > 800 else op_content_prod_raw
        accelerator_details_prompt_list.append(f"""
    - **Acelerador:** {product_name_original}
      - **Resumo do Battle Card (GCS):** {bc_summary if bc_summary else 'Não disponível.'}
      - **Detalhes do Data Sheet (GCS):** {ds_summary if ds_summary else 'Não disponível.'}
      - **Detalhes do Plano Operacional (GCS):** {op_summary if op_summary else 'Não disponível.'}
      - **Aplicação Específica no Órgão (Input do Usuário para {product_name_original}):** {user_integration_detail if user_integration_detail else 'Nenhum detalhe de integração fornecido.'}
        """)
    accelerator_details_prompt_section = "\n".join(accelerator_details_prompt_list) if accelerator_details_prompt_list else "Nenhum acelerador Xertica.ai selecionado."

    proposta_comercial_content = llm_context_data.get("proposta_comercial_content", "Conteúdo da proposta comercial não fornecido.")
    proposta_tecnica_content = llm_context_data.get("proposta_tecnica_content", "Conteúdo da proposta técnica não fornecido.")
    price_map_federal_template = """
| Tipo de Licença/Serviço | Fonte de Pesquisa/Contrato Referência | Valor Unitário Anual (R$) | Valor Mensal (R$) | Quantidade Referencial | Valor Total Estimado (R$) Anual |
|---|---|---|---|---|---|
| [Preencher] | [Preencher] | [Preencher] | [Preencher] | [Preencher] | [Preencher] |
"""[1:]
    price_map_estadual_municipal_template = """
| Tipo de Licença/Serviço | Fonte de Pesquisa/Contrato Referência | Empresa Contratada (Ref.) | Valor Unitário Anual (R$) | Valor Mensal (R$) | Quantidade Referencial | Valor Total Estimado (R$) Anual |
|---|---|---|---|---|---|---|
| [Preencher] | [Preencher] | Xertica.ai | [Preencher] | [Preencher] | [Preencher] | [Preencher] |
"""[1:]
    price_map_to_use_template = price_map_federal_template if esfera_administrativa == "Federal" else price_map_estadual_municipal_template
    produtos_originais_display_str = ', '.join([name_norm.replace('_', ' ') for name_norm in produtos_selecionados_normalizados]) if produtos_selecionados_normalizados else 'Nenhum acelerador especificado'
    
    abes_certs_str_parts = []
    for product_name, content in llm_context_data.get('gcs_abes_certificates_content', {}).items():
        if content:
            abes_certs_str_parts.append(f"Certificado ABES para {product_name}:\n{content}\n---\n")
    abes_certs_str = "\n".join(abes_certs_str_parts) if abes_certs_str_parts else "Nenhum certificado ABES carregado.\n"
    
    coe_content_str = llm_context_data.get('gcs_coe_content', "Conteúdo do Centro de Excelência não carregado.\n")

    # =======================================================================
    # PASSO 2: DEFINIÇÃO DO PROMPT COMPLETO E FINAL
    # Cole aqui o SEU prompt completo que você quer enviar para o Gemini,
    # usando todas as variáveis montadas acima.
    # =======================================================================
    llm_prompt_content_final = f"""
Você é um assistente de IA altamente especializado na elaboração de documentos técnicos e legais para o setor público brasileiro (esferas Federal, Estadual e Municipal), com expertise em licitações (Lei nº 14.133/2021, Lei 13.303/2016) e nas soluções de Inteligência Artificial da Xertica.ai.

Sua tarefa é gerar duas seções completas (ETP e TR) em Markdown.
**O conteúdo gerado DEVE ser em PROSA rica e detalhada, com análises aprofundadas, justificativas robustas e descrições técnicas claras.** Para listas (ex: requisitos, obrigações), use o formato de lista Markdown (`*` ou `-`).
Siga rigorosamente os modelos de estrutura ETP e TR fornecidos e adapte o conteúdo à esfera administrativa ({esfera_administrativa}) do `{orgao_nome}`.
Preencha todos os `[ ]` e `{{placeholders}}` nos modelos com informações contextualmente relevantes, gerando todo o texto dinâmico e analítico necessário.

**Sua resposta FINAL DEVE ser UM OBJETO JSON VÁLIDO.**
```json
{{
    "subject": "Título Descritivo do Documento (ETP e TR para {orgao_nome})",
    "etp_content": "Conteúdo COMPLETO do ETP em Markdown...",
    "tr_content": "Conteúdo COMPLETO do TR em Markdown..."
}}
Regras Detalhadas:

Adaptação por Esfera Administrativa: Ajuste linguagem e referências legais para {esfera_administrativa}.
Placeholders: Para campos como número de processo, nomes de responsáveis, use **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**. Processo: XXXXXX/{ano_atual}. CNAE: Sugira "6204-0/00 - Consultoria em tecnologia da informação", indicando que o órgão deve confirmar. Prazos de pagamento: Use valores comuns (ex: "5 dias úteis", "15 dias corridos").
Mapa de Preços: Preencha a tabela realisticamente. Se {valor_estimado_input} for fornecido, aproxime a soma.
Análise Profunda (RAG): Use rigorosamente os dados de GCS (Battle Cards, Data Sheets, Planos Operacionais), propostas PDF, certificados ABES e CoE para detalhar as soluções Xertica.ai. Destaque os diferenciais da Xertica.ai (integração GCP, especialização no setor público, inovação, etc.). Conecte as soluções à {justificativa_necessidade} e {objetivo_geral}.
Resultados Esperados: Detalhe com indicadores qualitativos e, se possível, quantitativos.
Justificativa Legal: Para {modelo_licitacao}, fundamente com a Lei 14.133/2021, Lei 13.303/2016 e contexto GCS, citando artigos.
Formato Markdown: Use #, ##, ###, *, -. Tabelas devem ser formatadas corretamente.
DADOS FORNECIDOS PELO USUÁRIO (Órgão Solicitante):

JSON

{json.dumps(llm_context_data, indent=2, ensure_ascii=False)}
CONTEÚDO EXTRAÍDO DAS PROPOSTAS XERTICA.AI (Anexos PDF):
Proposta Comercial: {proposta_comercial_content}
Proposta Técnica: {proposta_tecnica_content}

MAPA DE PREÇOS DE REFERÊNCIA (Estrutura Orientativa):
{price_map_to_use_template}

CONTEÚDO DE ACELERADORES XERTICA.AI (GCS - Battle Cards, Data Sheets, OP):
{gcs_accel_str}

CONTEÚDO DE DOCUMENTOS LEGAIS E CONTEXTO ADICIONAL (GCS):
{gcs_legal_str}

CONTEÚDO DE CERTIFICADOS ABES (GCS):
{abes_certs_str}

CONTEÚDO DO CENTRO DE EXCELÊNCIA XERTICA.AI (GCS):
{coe_content_str}

DETALHES DOS ACELERADORES (Input do Usuário e Contexto GCS):
{accelerator_details_prompt_section}

Mapeamento de Placeholders (Use estes para guiar o preenchimento):
{{sumario_aceleradores}}: "{produtos_originais_display_str}"
{{processo_administrativo_numero}}: "XXXXXX/{ano_atual}"
{{local_etp_full}}: "{local_etp_full_placeholder}"
{{mes_extenso}}: "{mes_extenso}"
{{ano_atual}}: "{ano_atual}"
{{esfera_administrativa}}: "{esfera_administrativa}"
{{orgao_nome}}: "{orgao_nome}"
{{titulo_projeto}}: "{titulo_projeto}"
{{justificativa_necessidade}}: "{justificativa_necessidade}"
{{objetivo_geral}}: "{objetivo_geral}"
{{modelo_licitacao}}: "{modelo_licitacao}"
{{contexto_geral_orgao}}: "{contexto_geral_orgao if contexto_geral_orgao else f'A {orgao_nome} busca modernizar seus serviços...'}"
{{valor_estimado_input_str}}: "{valor_estimado_input if valor_estimado_input is not None else '[VALOR NÃO FORNECIDO, ESTIMAR]'}"
{{prazos_estimados}}: "{prazos_estimados}"
{{parcelamento_contratacao}}: "{parcelamento_contratacao}"
{{justificativa_parcelamento_input}}: "{justificativa_parcelamento if justificativa_parcelamento else 'Não fornecida.'}"
{{produtos_originais_display_str}}: "{produtos_originais_display_str}"
{{introducao_etp}}: ... (Defina aqui o que o LLM deve gerar para este placeholder)
{{problema_necessidade}}: ...
{{referencia_in_sgd_me}}: ...
{{necessidades_negocio}}: ...
{{requisitos_tecnicos_funcionais}}: ...
{{levantamento_mercado}}: ...
{{estimativa_demanda}}: ...
{{mapa_comparativo_custos}}: ...
{{valor_estimado_total_etp}}: ...
{{descricao_solucao_etp}}: ...
{{parcelamento_justificativa}}: ...
{{providencias_tomadas}}: ...
{{declaracao_viabilidade}}: ...
{{nomes_cargos_responsaveis}}: ...
{{local_data_aprovacao}}: ...
{{cidade_uf_tr}}: "{local_etp_full_placeholder.split(',')[0]}" # Tenta extrair cidade/UF do local do ETP
{{data_tr}}: "{today.day} de {mes_extenso} de {ano_atual}"
{{numero_processo_administrativo_tr}}: "XXXXXX/{ano_atual}"
{{cnae_sugerido}}: "6204-0/00 - Consultoria em tecnologia da informação"
{{prazo_vigencia_tr}}: "{prazos_estimados}" # Ou um valor padrão como "12 meses"
{{regras_subcontratacao}}: ...
{{regras_garantia}}: ...
{{medicao_pagamento_dias_recebimento}}: "5 (cinco) dias úteis"
{{medicao_pagamento_dias_faturamento}}: "5 (cinco) dias úteis após o ateste" # Exemplo
{{medicao_pagamento_dias_pagamento}}: "15 (quinze) dias corridos após o ateste da Fatura"
{{criterio_julgamento_tr}}: ...
{{metodologia_implementacao}}: ...
{{criterios_aceitacao_tr}}: ...
{{obrigações_contratado_tr}}: ...
{{obrigações_orgao_tr}}: ...
{{gestao_contrato_tr}}: ...
{{sancoes_administrativas_tr}}: ...
{{anexos_tr}}: "Proposta Comercial e Técnica da Xertica.ai, Documentação dos Aceleradores (BC, DS, OP), Certificados ABES, Documento CoE, Exemplos Legais."

MODELO DE ETP PARA PREENCHIMENTO:

Estudo Técnico Preliminar
Contratação de solução tecnológica para {{titulo_projeto}}

Processo Administrativo nº {{processo_administrativo_numero}}

{{local_etp_full}}

Histórico de Revisões
Data	Versão	Descrição	Autor
{today.strftime('%d/%m/%Y')}	1.0	Finalização da primeira versão do documento	IA Xertica.ai

Exportar para as Planilhas
Área requisitante
Identificação da Área requisitante: {{orgao_nome}}
Nome do Responsável: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Matrícula: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]

Introdução
{{introducao_etp}}
O Estudo Técnico Preliminar – ETP é o documento constitutivo da primeira etapa do planejamento de uma contratação, que caracteriza o interesse público envolvido e a sua melhor solução. Ele serve de base ao Termo de Referência a ser elaborado, caso se conclua pela viabilidade da contratação.

O ETP tem por objetivo identificar e analisar os cenários para o atendimento de demanda registrada no Documento de Formalização da Demanda – DFD, bem como demonstrar a viabilidade técnica e econômica das soluções identificadas, fornecendo as informações necessárias para subsidiar a tomada de decisão e o prosseguimento do respectivo processo de contratação.
{{contexto_geral_orgao}}

Referência: Inciso XI, do art. 2º e art. 11 da IN SGD/ME nº 94/2022. {{referencia_in_sgd_me}}

Descrição do problema e das necessidade
{{problema_necessidade}}
A {{orgao_nome}} busca, por meio da iniciativa '{{titulo_projeto}}', endereçar um desafio crítico identificado: {{justificativa_necessidade}}. Este problema impede [descrever impacto negativo, ex: a eficiente prestação de serviços, a tomada de decisões ágeis, a otimização de recursos].

Necessidades do negócio
{{necessidades_negocio}}
A contratação visa suprir as seguintes necessidades do negócio do {{orgao_nome}}, com impactos diretos na eficiência operacional e na entrega de serviços:

Redução de gargalos operacionais: Eliminar pontos de lentidão e ineficiência nos processos atuais, permitindo um fluxo de trabalho mais dinâmico.
Melhoria da experiência do cidadão e do usuário interno: Proporcionar canais de comunicação e acesso a serviços mais intuitivos, rápidos e satisfatórios, elevando os índices de aprovação.
Otimização da alocação de recursos: Liberar equipes e colaboradores de tarefas repetitivas e de baixo valor agregado, permitindo que se concentrem em atividades estratégicas e de maior impacto.
Tomada de decisões baseada em dados: Fornecer dashboards e relatórios analíticos que transformem grandes volumes de dados em inteligência acionável, subsidiando escolhas estratégicas e operacionais.
Garantia de conformidade e transparência: Assegurar que as operações estejam em total alinhamento com a legislação vigente e com os princípios de publicidade, promovendo a confiança e a integridade.
Requisitos da Contratação
{{requisitos_tecnicos_funcionais}}
Os requisitos gerais e específicos para a contratação da solução proposta são:

Requisitos Funcionais: A solução deve ser capaz de {{objetivo_geral}}, atuando de forma a [descrever como as funcionalidades chaves da solução Xertica.ai se conectam ao objetivo]. Detalhar com base nos aceleradores: {{sumario_aceleradores}}.
Requisitos Não Funcionais:
Segurança: A solução deve garantir a integridade, confidencialidade e disponibilidade dos dados, em conformidade com a LGPD e as melhores práticas de segurança da informação para o setor público.
Escalabilidade: Deve possuir capacidade de expandir seus recursos e funcionalidades para atender a um aumento futuro na demanda e no volume de dados, sem degradação de desempenho.
Disponibilidade: A solução deve operar com alta disponibilidade, minimizando interrupções e garantindo acesso contínuo aos serviços, com SLA (Service Level Agreement) de no mínimo 99.5%.
Integração: Deve ter flexibilidade para integrar-se com os sistemas legados e plataformas já utilizadas pelo {{orgao_nome}}, utilizando APIs ou outros protocolos de comunicação padrão.
Desempenho: A solução deve apresentar tempo de resposta ágil e eficiente, mesmo em cenários de alta demanda, assegurando uma experiência fluida para o usuário.
Aderência Tecnológica: A solução deve estar alinhada com as especificações detalhadas na proposta técnica da Xertica.ai, que aborda aspectos de arquitetura, stack tecnológica e compatibilidade.
Levantamento de mercado
{{levantamento_mercado}}
O levantamento de mercado demonstrou que a contratação de soluções de Inteligência Artificial para otimização de processos e atendimento é uma tendência consolidada no setor público e privado. A Xertica.ai se destaca por sua notória especialização em projetos para o setor público brasileiro, comprovada por [Citar exemplos de sucesso ou diferenciais como Certificados ABES, menção ao CoE, parcerias como MTI/SERPRO se aplicável com base no contexto GCS].

A análise de mercado evidenciou que a Xertica.ai oferece um diferencial competitivo significativo em termos de integração nativa com o Google Cloud Platform (se aplicável), agilidade na implantação via aceleradores e uma compreensão aprofundada das particularidades legais e operacionais da administração pública.

Estimativa de demanda - quantidade de bens e serviços
{{estimativa_demanda}}
A estimativa de demanda para os serviços/bens objeto desta contratação será:

Quantitativos: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO, com base em volume esperado de usuários, transações ou dados]. Serão definidos em detalhe no Termo de Referência.
Mapa comparativo dos custos
{{mapa_comparativo_custos}}
O valor estimado foi ratificado por [CITE FONTES DE PESQUISA, PUBLICAÇÕES OU DISPENSA/INEXIGIBILIDADE SE PUDER OU DEIXE EM ABERTO]:
{price_map_to_use_template}

Estimativa de custo total da contratação
{{valor_estimado_total_etp}}
O valor estimado global para esta contratação, para o período de {{prazos_estimados}}, é de R$ {{valor_estimado_input_str}}.

Descrição da solução como um todo
{{descricao_solucao_etp}}
A solução proposta abrange a implementação dos aceleradores de Inteligência Artificial da Xertica.ai: {{sumario_aceleradores}}.

Justificativa do parcelamento ou não da contratação
{{parcelamento_justificativa}}
Decisão sobre Parcelamento: {{parcelamento_contratacao}}.
Justificativa: {{{{justificativa_parcelamento_input}} if parcelamento_contratacao == 'Justificar' and justificativa_parcelamento else f"A decisão por {('parcelar' if parcelamento_contratacao == 'Sim' else 'não parcelar')} a contratação foi embasada na busca por {('maior flexibilidade e entregas incrementais.' if parcelamento_contratacao == 'Sim' else 'garantir a integralidade da solução e sinergia entre componentes.')}"}}

Providências a serem tomadas
{{providencias_tomadas}}

Formalização do processo e assinatura do contrato.
Definição de cronograma detalhado.
Designação de equipe técnica do {{orgao_nome}}.
Capacitação e transferência de conhecimento.
Configuração e personalização da solução Xertica.ai.
Implantação em produção e acompanhamento.
Acompanhamento contínuo da performance.
Declaração de viabilidade
{{declaracao_viabilidade}}
Declara-se a viabilidade plena da contratação da solução Xertica.ai.

Responsáveis
{{nomes_cargos_responsaveis}}
Equipe de Planejamento da Contratação (Portaria nº [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO], de {today.day} de {mes_extenso} de {ano_atual}).
INTEGRANTE TÉCNICO: [Nome, Matrícula/SIAPE]
INTEGRANTE REQUISITANTE: [Nome, Matrícula/SIAPE]

Aprovação e declaração de conformidade
Aprovo este Estudo Técnico Preliminar.
{{local_etp_full}} {{local_data_aprovacao}}
&lt;NEWPAGE>
MODELO DE TR PARA PREENCHIMENTO:

Termo de Referência – Nº [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
1 – DEFINIÇÃO DO OBJETO
{{sumario_aceleradores}}
Contratação de solução de {{sumario_aceleradores}} da Xertica.ai para {{objetivo_geral}} no {{orgao_nome}}.

1.1. Objeto Sintético: Contratação de serviços de TI com IA, caracterizados como [SERVIÇOS TÉCNICOS ESPECIALIZADOS DE CONSULTORIA E DESENVOLVIMENTO DE IA], para modernização da Administração Pública {{esfera_administrativa}}.

Ramo de Atividade: {{cnae_sugerido}} (o CNAE específico deverá ser confirmado pelo órgão).
Quantitativos estimados: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO].
Prazo do contrato: {{prazo_vigencia_tr}}.
2 – FUNDAMENTAÇÃO DA CONTRATAÇÃO
Detalhada nos Estudos Técnicos Preliminares (ETP) anexos.
2.1. Previsto no Plano de Contratações Anual {ano_atual} do {{orgao_nome}}:

ID PCA no PNCP e/ou SGA: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Data de publicação no PNCP: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Id do item no PCA: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO] 2.2. Justificativa: {{justificativa_necessidade}} no {{orgao_nome}}. 2.3. Enquadramento: {{modelo_licitacao}}, Lei nº 14.133/2021. [SE INEXIGIBILIDADE/DISPENSA, CITAR ARTIGOS].
3 – DESCRIÇÃO DA SOLUÇÃO COMO UM TODO
Implementação e suporte dos aceleradores Xertica.ai: {{sumario_aceleradores}}.
3.1. Compreende: Disponibilização, instalação, configuração, serviços de implementação, customização, integração, suporte técnico, manutenção, treinamento.
3.2. Forma de execução: indireta, regime de [EMPREITADA POR PREÇO GLOBAL/ESCOPO - A SER DEFINIDO PELO ÓRGÃO].
3.3. Detalhes no ETP e Proposta Técnica Xertica.ai.

4 – REQUISITOS DA CONTRATAÇÃO
4.1. Requisitos:

Funcionais: Atender a {{objetivo_geral}} com funcionalidades dos aceleradores. [CITE 3-5 FUNCIONALIDADES CHAVES].
Não Funcionais: Segurança (LGPD), Escalabilidade, Disponibilidade (SLA 99.5%), Integração, Desempenho, Manutenibilidade.
Práticas de Sustentabilidade: Conforme Lei nº 14.133/2021. 4.2. Carta de solidariedade: [SIM/NÃO - A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]. 4.3. SUBCONTRATAÇÃO: {{regras_subcontratacao}} (Ex: Não permitida para partes relevantes). 4.4. GARANTIA DA CONTRATAÇÃO: {{regras_garantia}} (Ex: 12 meses). 4.5. Transição contratual com transferência de conhecimento. 4.6. VISTORIA: [SIM/NÃO - A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO].
5 – EXECUÇÃO DO OBJETO
5.1. Prazos conforme Ordem de Serviço e cronograma da Proposta Técnica.
5.2. Execução predominantemente remota, visitas técnicas se acordado.
5.3. Observar métodos da Proposta Técnica Xertica.ai. {{metodologia_implementacao}}
5.4. CONTRATADA fornecerá todos os recursos necessários.
5.5. Prazo de garantia contratual: mínimo 12 meses (Art. 99, Lei nº 14.133/2021).

6 – GESTÃO DO CONTRATO
{{gestao_contrato_tr}}
Fiscal do {{orgao_nome}} acompanhará a execução.
6.1. Execução fiel ao contrato e Lei nº 14.133/2021.
6.2. Comunicações por escrito.
6.3. CONTRATANTE pode convocar representante da empresa.
6.4. Contrato publicado no PNCP e portal de transparência.
6.5. Reunião inicial para alinhamento e plano de fiscalização.
6.6. Acompanhamento contínuo por fiscal(is) designados.
6.7. CONTRATADA manterá preposto.

7 – MEDIÇÃO E PAGAMENTO
7.1. Avaliação por Instrumento de Medição de Resultado (IMR): [DEFINIR METODOLOGIA E REGRAS - Ex: SLAs, disponibilidade, satisfação]. {{criterios_aceitacao_tr}}
7.2. Pagamento mensal conforme indicadores do IMR.
7.3. Recebimento provisório e definitivo (Art. 140, Lei nº 14.133/2021).
7.4. Faturamento: Fatura/Nota Fiscal em {{medicao_pagamento_dias_recebimento}} após ateste da medição, com regularidade fiscal/trabalhista.
7.5. Pagamento: {{medicao_pagamento_dias_pagamento}} após atesto da Fatura/Nota Fiscal. (Prazo para faturamento pela contratada: {{medicao_pagamento_dias_faturamento}})

8 – SELEÇÃO DO FORNECEDOR
8.1. Seleção por {{modelo_licitacao}}, critério de julgamento: {{criterio_julgamento_tr}} [MENOR PREÇO/MELHOR TÉCNICA/TÉCNICA E PREÇO/MAIOR RETORNO ECONÔMICO - A SER DEFINIDO PELO ÓRGÃO].

Exigências de Habilitação: Conforme Art. 62-70, Lei nº 14.133/2021 (jurídica, técnica, econômico-financeira, fiscal/trabalhista, cumprimento do inciso XXXIII, art. 7º CF).
9 – ESTIMATIVA DO PREÇO
9.1. Valor estimado: R$ {{valor_estimado_input_str}}, detalhado no ETP.

10 – ADEQUAÇÃO ORÇAMENTÁRIA
Recursos da Lei Orçamentária Anual do(a) {{esfera_administrativa}} ({{orgao_nome}}).
10.1. Dotação:

UG Executora: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Programa de Trabalho: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Fonte: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Natureza da Despesa: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - ex: 33.90.39 ou 44.90.39] 10.2. Dotação de exercícios subsequentes indicada após aprovação da LOA respectiva.
{{anexos_tr}}
Há anexos no pedido: Sim (Proposta Comercial e Técnica da Xertica.ai, etc.).

OBRIGAÇÕES DO CONTRATADO: {{obrigações_contratado_tr}}
OBRIGAÇÕES DO ÓRGÃO: {{obrigações_orgao_tr}}
SANÇÕES ADMINISTRATIVAS: {{sancoes_administrativas_tr}}

Gere o objeto JSON agora.
"""

    # =======================================================================
    # PASSO 3: CHAMADA À API GEMINI E PROCESSAMENTO DA RESPOSTA (UM ÚNICO BLOCO TRY/EXCEPT)
    # =======================================================================
    response_text = None
    try:
        logger.info(f"Enviando prompt para o Gemini (primeiros 1000 chars): {llm_prompt_content_final[:1000].replace('\n', ' ')}...")
        
        response = await gemini_model.generate_content_async(
            llm_prompt_content_final,
            generation_config=_generation_config
        )
        
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            logger.error(f"Resposta do Gemini inválida ou sem conteúdo esperado. Resposta completa: {response}")
            raise Exception("Resposta inválida do modelo Gemini (sem partes de conteúdo).")

        response_text = response.candidates[0].content.parts[0].text
        logger.info(f"Resposta RAW do Gemini recebida (primeiros 500 chars): {response_text[:500].replace('\n', ' ')}...")
        
        match_json = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
        if match_json:
            json_str = match_json.group(1)
            logger.info("JSON extraído de bloco de código Markdown.")
        else:
            json_str = response_text
            logger.info("Resposta do Gemini assumida como JSON direto (sem bloco de código Markdown).")
            
        parsed_content = json.loads(json_str)
        
        logger.info(f"Conteúdo parseado do Gemini. Tipo: {type(parsed_content)}")
        if isinstance(parsed_content, dict):
            logger.info(f"Chaves do dicionário parseado: {list(parsed_content.keys())}")
        else:
            logger.error(f"ALERTA CRÍTICO: Conteúdo parseado do Gemini NÃO é um dicionário! Conteúdo (primeiros 500 chars): {str(parsed_content)[:500]}")
            raise ValueError(f"LLM_OUTPUT_FORMAT_ERROR: Esperava um objeto JSON (dict), mas recebi {type(parsed_content)}. Verifique a resposta do LLM.")
            
        logger.info("Resposta do Gemini parseada como JSON e validada como dict com sucesso.")
        return parsed_content

    except json.JSONDecodeError as e:
        logger.error(f"Erro ao parsear JSON da resposta do Gemini: {e}.")
        problematic_json_string = response_text if response_text is not None else "String JSON não capturada."
        logger.error(f"String JSON que causou o erro (primeiros 1000 chars): {problematic_json_string[:1000]}")
        raise HTTPException(status_code=500, detail=f"Erro no formato JSON retornado pelo Gemini: {e}. Verifique os logs do servidor para a string exata.")
    except AttributeError as e:
        response_str_for_log = str(response)[:500] if 'response' in locals() and response is not None else "Response object not available or None."
        logger.error(f"Estrutura da resposta do Gemini inesperada: {e}. Resposta (início): {response_str_for_log}")
        raise HTTPException(status_code=500, detail=f"Formato de resposta inesperado do Gemini: {e}")
    except Exception as e:
        logger.exception(f"Erro crítico ao chamar a API do Gemini ou processar sua resposta: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na geração de conteúdo via IA: {e}")
               
@app.post("/generate_etp_tr", summary="Gera Documentos ETP e TR", tags=["Documentos"])
async def generate_etp_tr_endpoint(
    request: Request,
    orgaoSolicitante: str = Form(..., description="Nome completo do órgão público solicitante."),
    tituloProjeto: str = Form(..., description="Título ou nome do projeto/contratação."),
    justificativaNecessidade: str = Form(..., description="Descrição detalhada do problema ou necessidade a ser resolvida."),
    objetivoGeral: str = Form(..., description="Objetivo principal da contratação/solução."),
    prazosEstimados: str = Form(..., description="Prazos estimados. Ex: 3 meses implantação, 12 meses operação."),
    modeloLicitacao: str = Form(..., description="Modelo de licitação (ex: Pregão Eletrônico, Inexigibilidade)."),
    parcelamentoContratacao: str = Form(..., description="Contratação parcelada (Sim, Não, Justificar)."),
    contextoGeralOrgao: Optional[str] = Form(None, description="Breve contexto sobre o órgão."),
    valorEstimado: Optional[float] = Form(None, description="Valor total estimado da contratação (opcional)."),
    justificativaParcelamento: Optional[str] = Form(None, description="Justificativa para parcelamento."),
    propostaComercialFile: Optional[UploadFile] = File(None, description="Proposta Comercial PDF (opcional)."),
    propostaTecnicaFile: Optional[UploadFile] = File(None, description="Proposta Técnica PDF (opcional).")
):
    logger.info(f"Requisição para gerar ETP/TR para '{tituloProjeto}' do órgão '{orgaoSolicitante}'.")
    if not gemini_model or not storage_client:
        raise HTTPException(status_code=503, detail="Serviços essenciais de IA ou Armazenamento não estão disponíveis.")

    form_data = await request.form()
    produtosXertica_list_normalized = form_data.getlist("produtosXertica")
    logger.info(f"Produtos Xertica selecionados (normalizados pelo frontend): {produtosXertica_list_normalized}")

    llm_context_data = {
        "orgaoSolicitante": orgaoSolicitante, "tituloProjeto": tituloProjeto,
        "justificativaNecessidade": justificativaNecessidade, "objetivoGeral": objetivoGeral,
        "prazosEstimados": prazosEstimados, "modeloLicitacao": modeloLicitacao,
        "parcelamentoContratacao": parcelamentoContratacao,
        "contextoGeralOrgao": contextoGeralOrgao or "Não fornecido.",
        "valorEstimado": valorEstimado,
        "justificativaParcelamento": justificativaParcelamento or "Não fornecida.",
        "produtosXertica": produtosXertica_list_normalized,
        "data_geracao_documento": date.today().strftime("%d/%m/%Y"),
        'gcs_accelerator_content': {}, 'gcs_legal_context_content': {},
        'gcs_abes_certificates_content': {}, 'gcs_coe_content': None
    }
    for product_name_normalized in produtosXertica_list_normalized:
        integration_key = f"integracao_{product_name_normalized}"
        llm_context_data[integration_key] = form_data.get(integration_key, f"Detalhes de integração para {product_name_normalized.replace('_', ' ')} não fornecidos.")

    if propostaComercialFile and propostaComercialFile.filename:
        logger.info(f"Processando Proposta Comercial: {propostaComercialFile.filename}")
        llm_context_data["proposta_comercial_content"] = await extract_text_from_pdf(propostaComercialFile)
        gcs_path_com = await upload_file_to_gcs(propostaComercialFile, f"propostas_clientes/{orgaoSolicitante.replace(' ','_')}_{tituloProjeto.replace(' ','_')}_comercial_{date.today().strftime('%Y%m%d')}_{propostaComercialFile.filename}")
        llm_context_data["commercial_proposal_gcs_uri"] = gcs_path_com
    else:
        llm_context_data["proposta_comercial_content"] = "Nenhuma proposta comercial em PDF foi fornecida pelo usuário."
        logger.info("Nenhum arquivo de proposta comercial fornecido.")

    if propostaTecnicaFile and propostaTecnicaFile.filename:
        logger.info(f"Processando Proposta Técnica: {propostaTecnicaFile.filename}")
        llm_context_data["proposta_tecnica_content"] = await extract_text_from_pdf(propostaTecnicaFile)
        gcs_path_tec = await upload_file_to_gcs(propostaTecnicaFile, f"propostas_clientes/{orgaoSolicitante.replace(' ','_')}_{tituloProjeto.replace(' ','_')}_tecnica_{date.today().strftime('%Y%m%d')}_{propostaTecnicaFile.filename}")
        llm_context_data["technical_proposal_gcs_uri"] = gcs_path_tec
    else:
        llm_context_data["proposta_tecnica_content"] = "Nenhuma proposta técnica em PDF foi fornecida pelo usuário."
        logger.info("Nenhum arquivo de proposta técnica fornecido.")

    for product_name_normalized in produtosXertica_list_normalized:
        product_original_name = product_name_normalized.replace('_', ' ')
        product_folder_name = product_original_name
        doc_types_map = {"BC": ["BC - ", "BC_", "BATTLE CARD DE "],"DS": ["DS - ", "DS_"],"OP": ["OP - ", "OP_"]}
        for doc_type_key, prefixes in doc_types_map.items():
            found_content = None
            for prefix in prefixes:
                path1 = f"{product_folder_name}/{prefix}{product_original_name}.txt"
                path2 = f"{product_folder_name}/{prefix}{product_name_normalized}.txt"
                path3 = f"{product_folder_name}/{prefix}{product_folder_name}.txt"
                path4_ds_upper = f"{product_folder_name}/{prefix}{product_original_name.upper()}.txt"
                paths_to_try = [path1, path2, path3]
                if doc_type_key == "DS": paths_to_try.append(path4_ds_upper)
                for path_attempt in paths_to_try:
                    content = get_gcs_file_content(path_attempt)
                    if content: found_content = content; break
                if found_content: break
            if found_content:
                llm_context_data['gcs_accelerator_content'][f"{product_original_name} ({doc_type_key})"] = found_content
            else:
                alt_bc_path = f"aceleradores_conteudo/{product_name_normalized}/BC_{product_name_normalized}.txt"
                alt_ds_path = f"aceleradores_conteudo/{product_name_normalized}/DS_{product_name_normalized}.txt"
                alt_op_path = f"aceleradores_conteudo/{product_name_normalized}/OP_{product_name_normalized}.txt"
                if doc_type_key == "BC" and get_gcs_file_content(alt_bc_path): llm_context_data['gcs_accelerator_content'][f"{product_original_name} ({doc_type_key})"] = get_gcs_file_content(alt_bc_path)
                elif doc_type_key == "DS" and get_gcs_file_content(alt_ds_path): llm_context_data['gcs_accelerator_content'][f"{product_original_name} ({doc_type_key})"] = get_gcs_file_content(alt_ds_path)
                elif doc_type_key == "OP" and get_gcs_file_content(alt_op_path): llm_context_data['gcs_accelerator_content'][f"{product_original_name} ({doc_type_key})"] = get_gcs_file_content(alt_op_path)
                else:
                    logger.warning(f"Documento {doc_type_key} para '{product_original_name}' não encontrado após várias tentativas.")
                    llm_context_data['gcs_accelerator_content'][f"{product_original_name} ({doc_type_key})"] = f"Conteúdo {doc_type_key} não encontrado."
        abes_path_options = [f"Certificados ABES/[Declaração ABES] ({product_original_name}).txt", f"Certificados ABES/[Declaração ABES] {product_original_name}.txt"]
        abes_content = None
        for abes_path in abes_path_options:
            abes_content = get_gcs_file_content(abes_path)
            if abes_content:
                llm_context_data['gcs_abes_certificates_content'][product_original_name] = abes_content
                logger.info(f"Certificado ABES para '{product_original_name}' carregado de {abes_path}.")
                break
        if not abes_content: logger.warning(f"Certificado ABES para '{product_original_name}' não encontrado.")

    gcp_analysis_content = get_gcs_file_content("GCP/Análise Técnica_ Google Cloud Platform_.txt")
    if gcp_analysis_content: llm_context_data['gcs_legal_context_content']["Análise Técnica GCP"] = gcp_analysis_content
    gmp_analysis_content = get_gcs_file_content("GMP/Google Maps Platform_ Análise Técnica_.txt")
    if gmp_analysis_content: llm_context_data['gcs_legal_context_content']["Análise Técnica GMP"] = gmp_analysis_content
    gws_analysis_content = get_gcs_file_content("GWS/Análise técnica do Google Workspace_.txt")
    if gws_analysis_content: llm_context_data['gcs_legal_context_content']["Análise Técnica GWS"] = gws_analysis_content
    coe_content = get_gcs_file_content("CoE/Centro de Excelência.txt")
    if coe_content: llm_context_data['gcs_coe_content'] = coe_content; logger.info("Documento CoE carregado.")
    else: logger.warning("Documento CoE não encontrado.")
    legal_docs_map = {
        "CONTRATO MTI XERTICA (Exemplo)": "Formas ágeis de contratação/MTI/CONTRATO DE PARCERIA 03-2024-MTI - XERTICA - ASSINADO.txt",
        "ATA REGISTRO PREÇOS MPAP XERTICA (Exemplo)": "Formas ágeis de contratação/MPAP/ATA DE REGISTRO DE PREÇOS Nº 041-2024-XERTICA.txt",
        "MOU SERPRO XERTICA (Exemplo)": "Formas ágeis de contratação/Serpro/[Xertica & Serpro] Memorando de Entendimento (MoU) - VersãoFinal.txt",
        "DETECÇÃO E ANÁLISE DE RISCOS (Contexto)": "Detecção e Análise de Riscos/Detecção de análise de riscos.txt",
        "CATÁLOGO GERAL SERVIÇOS IA MTI (Contexto)": "Formas ágeis de contratação/MTI/Catalogo_Geral_de_Servicos_de_Inteligencia_Artificial_-_CGSIA._Versao_Final_1-_ASSINADO.txt",
        "MANUAL MTI.IA XERTICA (Contexto)": "Formas ágeis de contratação/MTI/MNG_-_Solucao_MTI.IA_-_XERTICA._Versao_Final_1_ASSINADO.txt"
    }
    for display_name, gcs_path in legal_docs_map.items():
        content = get_gcs_file_content(gcs_path)
        if content: llm_context_data['gcs_legal_context_content'][display_name] = content
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logger.debug(f"Dados completos de contexto para LLM (sem conteúdo de arquivos): {{key: (type(value), len(value) if isinstance(value, str) else 'N/A') for key, value in llm_context_data.items()}}")

    llm_response = await generate_etp_tr_content_with_gemini(llm_context_data)
    document_subject = llm_response.get("subject", f"ETP e TR: {orgaoSolicitante} - {tituloProjeto} ({date.today().strftime('%Y-%m-%d')})")
    etp_content_md = llm_response.get("etp_content", "# ETP\n\nErro: Conteúdo do ETP não foi gerado corretamente pelo LLM.")
    tr_content_md = llm_response.get("tr_content", "# Termo de Referência\n\nErro: Conteúdo do TR não foi gerado corretamente pelo LLM.")

    docs_service, drive_service = authenticate_google_docs_and_drive()
    if not docs_service or not drive_service:
        raise HTTPException(status_code=503, detail="Falha na autenticação com Google Docs/Drive API. Verifique permissões da Service Account.")
    try:
        new_doc_body = {'title': document_subject}
        new_doc_metadata = drive_service.files().create(body=new_doc_body, fields='id,webViewLink').execute()
        document_id = new_doc_metadata.get('id')
        document_link_initial = new_doc_metadata.get('webViewLink')
        if not document_id:
            logger.error("Falha ao criar novo documento no Google Docs. ID não retornado.")
            raise HTTPException(status_code=500, detail="Falha ao criar novo documento no Google Docs (ID não obtido).")
        logger.info(f"Documento Google Docs criado com ID: {document_id}, Link inicial: {document_link_initial}")
        combined_markdown_content = f"{etp_content_md}\n<NEWPAGE>\n{tr_content_md}"
        requests_for_docs_api = apply_basic_markdown_to_docs_requests(combined_markdown_content)
        if requests_for_docs_api:
            MAX_REQUESTS_PER_BATCH = 400
            for i in range(0, len(requests_for_docs_api), MAX_REQUESTS_PER_BATCH):
                batch = requests_for_docs_api[i:i + MAX_REQUESTS_PER_BATCH]
                docs_service.documents().batchUpdate(documentId=document_id,body={'requests': batch}).execute()
                logger.info(f"Lote de {len(batch)} requests enviado para Google Docs API (documento: {document_id}).")
            logger.info(f"Conteúdo ETP e TR inserido e formatado no documento Google Docs: {document_id}")
        else:
            logger.warning(f"Nenhuma request de formatação gerada para o documento {document_id}.")
        permission_role = 'reader'
        permission = {'type': 'anyone', 'role': permission_role}
        try:
            drive_service.permissions().create(fileId=document_id, body=permission, fields='id').execute()
            logger.info(f"Permissões de '{permission_role}' públicas definidas para o documento: {document_id}")
        except HttpError as e_perm:
            logger.warning(f"Não foi possível aplicar permissão '{permission_role}' ao documento {document_id}: {e_perm}. O documento pode não ser publicamente acessível.")
        document_link_final = document_link_initial
        if not document_link_final:
            file_metadata_final = drive_service.files().get(fileId=document_id, fields='webViewLink').execute()
            document_link_final = file_metadata_final.get('webViewLink')
            if not document_link_final: document_link_final = f"https://docs.google.com/document/d/{document_id}/edit"
        logger.info(f"Processo de geração de ETP/TR concluído com sucesso. Link do Documento: {document_link_final}")
        return JSONResponse(status_code=200, content={
            "success": True, "message": "Documentos ETP e TR gerados e salvos no Google Docs.",
            "doc_link": document_link_final, "document_id": document_id,
            "commercial_proposal_gcs_uri": llm_context_data.get("commercial_proposal_gcs_uri"),
            "technical_proposal_gcs_uri": llm_context_data.get("technical_proposal_gcs_uri")
        })
    except HttpError as e_google_api:
        error_message = f"Erro na API do Google. Status: {e_google_api.resp.status}"
        try:
            error_content = e_google_api.content.decode()
            error_details_json = json.loads(error_content)
            error_message = error_details_json.get('error', {}).get('message', error_message)
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
            logger.warning(f"Não foi possível decodificar ou parsear detalhes do erro da API do Google: {getattr(e_google_api, 'content', 'N/A')}")
        logger.exception(f"Erro na API do Google Docs/Drive: {error_message}")
        raise HTTPException(status_code=e_google_api.resp.status if hasattr(e_google_api, 'resp') else 500, detail=f"Erro na API do Google Docs/Drive: {error_message}")
    except Exception as e_general:
        logger.exception(f"Erro inesperado durante a geração ou criação do documento Google Docs: {e_general}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e_general}. Verifique os logs.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
