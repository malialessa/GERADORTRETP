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

    logger.info("Iniciando chamada ao modelo Gemini para geração de ETP/TR.")
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
    local_etp_full_placeholder = f"[LOCAL PADRÃO - CIDADE/UF], {today.day} de {mes_extenso} de {ano_atual}"
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
      - **Aplicação Específica no Órgão (Input do Usuário para {product_name_original}):** {user_integration_detail if user_integration_detail else 'Nenhum detalhe de integração fornecido. O LLM deve inferir a aplicação com base no problema/solução, nos documentos do acelerador e no contexto Xertica.ai'}
        """)
    accelerator_details_prompt_section = "\n".join(accelerator_details_prompt_list) if accelerator_details_prompt_list else "Nenhum acelerador Xertica.ai selecionado ou detalhes não fornecidos."
    proposta_comercial_content = llm_context_data.get("proposta_comercial_content", "Conteúdo da proposta comercial não fornecido ou erro na extração.")
    proposta_tecnica_content = llm_context_data.get("proposta_tecnica_content", "Conteúdo da proposta técnica não fornecido ou erro na extração.")
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

    # ==========================================================================
    # INÍCIO DO PLACEHOLDER PARA O CONTEÚDO DO PROMPT
    # ==========================================================================
    llm_prompt_content_final = f"""# COLE AQUI O CONTEÚDO COMPLETO DO SEU PROMPT DEFINIDO ANTERIORMENTE
# Certifique-se de que as variáveis como {json.dumps(llm_context_data, indent=2, ensure_ascii=False)},
# {proposta_comercial_content}, {proposta_tecnica_content}, {price_map_to_use_template},
# {gcs_accel_str}, {gcs_legal_str}, {abes_certs_str}, {coe_content_str},
# {accelerator_details_prompt_section}, e todos os placeholders {{exemplo}}
# estão corretamente formatados dentro da f-string.
"""
    # ==========================================================================
    # FIM DO PLACEHOLDER PARA O CONTEÚDO DO PROMPT
    # ==========================================================================

    response_text = None
    try:
        logger.info(f"Enviando prompt para o Gemini (primeiros 1000 chars): {llm_prompt_content_final[:1000].replace('\n', ' ')}...")
        response = await gemini_model.generate_content_async( # Usando async conforme original
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
        logger.error(f"Estrutura da resposta do Gemini inesperada: {e}. Resposta: {str(response)[:500]}")
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
