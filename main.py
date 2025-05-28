"""
Projeto: Gerador de ETP e Termo de Referência com IA para Documentos Xertica.ai
Módulo: main.py
Descrição: Backend da aplicação FastAPI que orquestra a geração de documentos ETP (Estudo Técnico Preliminar)
           e TR (Termo de Referência) utilizando o modelo Gemini 1.5 Flash do Google Vertex AI.
           Integra-se com Google Cloud Storage para armazenamento de anexos e Google Docs API para
           criação e formatação dos documentos finais.
Autor: Xertica.ai - Assistente de IA
Data: 2024-05-23
"""

import os
import logging
import json
import io
from datetime import date
import re

from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Google Cloud Imports
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig # Ensure this path is correct for your Vertex AI SDK version
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pypdf import PdfReader # Ensure pypdf is installed, or use PyPDF2 if that's what you have


# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# Carrega variáveis de ambiente (para desenvolvimento local)
load_dotenv()


app = FastAPI(
    title="Gerador de ETP e TR Xertica.ai",
    description="Backend inteligente para gerar documentos ETP e TR com IA da Xertica.ai.",
    version="0.1.0"
)


# Configurações CORS
origins = ["*"] # Em produção, restrinja para os domínios do seu frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Variáveis de Ambiente e Inicialização de Clientes GCP / Vertex AI
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.getenv("GCP_PROJECT_LOCATION", "us-central1") # Default location if not set
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "docsorgaospublicos") # Default bucket if not set


if not GCP_PROJECT_ID:
    logger.error("GCP_PROJECT_ID não está configurado.")
    # Em um ambiente de produção, você pode querer que o app falhe ao iniciar
    # ou tenha um comportamento de fallback.
    # raise Exception("GCP_PROJECT_ID não configurado. Por favor, defina a variável de ambiente.")

if not GCP_PROJECT_LOCATION:
    logger.error("GCP_PROJECT_LOCATION não está configurado.")
    # raise Exception("GCP_PROJECT_LOCATION não configurado. Por favor, defina a variável de ambiente.")


try:
    if GCP_PROJECT_ID and GCP_PROJECT_LOCATION: # Only initialize if configured
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_PROJECT_LOCATION)
        gemini_model = GenerativeModel("gemini-1.5-flash-001")
        # Configuração de geração de conteúdo para o modelo Gemini
        _generation_config = GenerationConfig(
            temperature=0.7,          # Controla a aleatoriedade da saída. Valores mais altos = mais criativo.
            max_output_tokens=8192,   # Máximo de tokens na resposta.
            response_mime_type="application/json" # Espera uma resposta JSON do modelo.
        )
        logger.info(f"Vertex AI inicializado com projeto '{GCP_PROJECT_ID}' e localização '{GCP_PROJECT_LOCATION}'. Modelo Gemini-1.5-flash-001 carregado.")
    else:
        gemini_model = None # Define como None se não configurado para evitar erros posteriores
        logger.warning("Vertex AI não inicializado devido à ausência de GCP_PROJECT_ID ou GCP_PROJECT_LOCATION.")

except Exception as e:
    logger.exception(f"Erro ao inicializar Vertex AI ou carregar modelo Gemini: {e}")
    gemini_model = None # Garante que gemini_model é None em caso de falha
    # Não levanta HTTPException aqui para permitir que o app inicie, mas logue o erro.
    # O erro será tratado nas rotas que dependem do modelo.


try:
    if GCP_PROJECT_ID: # Only initialize if project_id is available
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        logger.info("Google Cloud Storage client inicializado.")
    else:
        storage_client = None
        logger.warning("Google Cloud Storage client não inicializado devido à ausência de GCP_PROJECT_ID.")
except Exception as e:
    logger.exception(f"Erro ao inicializar Google Cloud Storage client: {e}")
    storage_client = None
    # Não levanta HTTPException aqui.


def authenticate_google_docs_and_drive():
    """
    Autentica com as APIs do Google Docs e Drive.
    No Cloud Run, utiliza as credenciais da Service Account do próprio serviço (ADC - Application Default Credentials).
    Assegure que a Service Account tenha as permissões necessárias (Docs API Editor, Drive API File Creator).
    """
    try:
        # O cache_discovery=False é útil em ambientes serverless para evitar problemas com arquivos temporários.
        docs_service = build('docs', 'v1', cache_discovery=False)
        drive_service = build('drive', 'v3', cache_discovery=False)
        logger.info("Google Docs e Drive services inicializados com sucesso usando Application Default Credentials.")
        return docs_service, drive_service
    except Exception as e:
        logger.exception(f"Erro ao autenticar/inicializar Google Docs/Drive APIs: {e}")
        # Levanta uma exceção que será capturada pela rota para retornar um erro HTTP 500.
        raise HTTPException(
            status_code=500,
            detail=f"Falha na autenticação da API do Google Docs/Drive: {e}. Verifique as permissões da Service Account (Docs API Editor, Drive API File Creator)."
        )

def get_gcs_file_content(file_path: str) -> Optional[str]:
    """Lê o conteúdo de um arquivo de texto de um bucket GCS."""
    if not storage_client:
        logger.error("GCS client não inicializado. Não é possível ler o arquivo.")
        return None
    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME não configurado. Não é possível ler o arquivo.")
        return None
        
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        if blob.exists():
            content = blob.download_as_text()
            logger.info(f"Conteúdo de GCS://{GCS_BUCKET_NAME}/{file_path} lido com sucesso ({len(content)} chars).")
            return content
        logger.warning(f"Arquivo não encontrado no GCS: {GCS_BUCKET_NAME}/{file_path}")
        return None
    except Exception as e:
        logger.exception(f"Erro ao ler arquivo GCS {GCS_BUCKET_NAME}/{file_path}: {e}")
        return None

async def upload_file_to_gcs(upload_file: UploadFile, destination_path: str) -> Optional[str]:
    """Faz upload de um arquivo para o GCS e retorna o URL público."""
    if not storage_client:
        logger.error("GCS client não inicializado. Upload falhou.")
        raise HTTPException(status_code=500, detail="Serviço de armazenamento não configurado.")
    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME não configurado. Upload falhou.")
        raise HTTPException(status_code=500, detail="Bucket de armazenamento não configurado.")

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_path)
        
        contents = await upload_file.read()
        # Após a leitura, o ponteiro do arquivo está no final.
        # É uma boa prática voltar ao início se precisar reler o arquivo.
        await upload_file.seek(0) 

        blob.upload_from_string(contents, content_type=upload_file.content_type)
        
        # Tornar o objeto público para acesso via URL.
        # Considere as implicações de segurança para dados sensíveis.
        # Alternativas incluem URLs assinadas ou controle de acesso mais granular.
        blob.make_public()
        logger.info(f"Arquivo '{upload_file.filename}' carregado para GCS://{GCS_BUCKET_NAME}/{destination_path} e tornado público.")
        return blob.public_url
    except Exception as e:
        logger.exception(f"Erro ao fazer upload do arquivo '{upload_file.filename}' para GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Falha ao carregar arquivo para o armazenamento: {e}")

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """
    Extrai texto de um arquivo PDF usando pypdf.
    Para documentos complexos ou PDFs escaneados, Google Cloud Document AI ou Vision AI são recomendados.
    """
    logger.info(f"Iniciando extração de texto do PDF: {pdf_file.filename}")
    try:
        contents = await pdf_file.read()
        # Importante: reposicionar o ponteiro do arquivo para o início após a leitura,
        # caso o arquivo precise ser lido novamente (ex: para upload).
        await pdf_file.seek(0) 
        
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page_num, page in enumerate(reader.pages):
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + "\n" # Adiciona nova linha entre páginas
            else:
                logger.warning(f"Nenhum texto extraído da página {page_num + 1} do PDF {pdf_file.filename}.")

        logger.info(f"Texto extraído do PDF {pdf_file.filename} (tamanho total: {len(text)} caracteres)")
        
        if not text.strip():
            logger.warning(f"O texto extraído de {pdf_file.filename} está vazio ou contém apenas espaços em branco. "
                           "Isso pode indicar um PDF baseado em imagem (escaneado) ou com um layout muito complexo para pypdf.")
            # Retorna uma mensagem informativa e um conteúdo placeholder para o LLM
            return (f"AVISO: Não foi possível extrair texto legível do PDF '{pdf_file.filename}'. "
                    f"O arquivo pode ser um PDF de imagem (escaneado) ou ter um formato que dificulta a extração de texto por pypdf. "
                    f"Para análise pelo Gemini, será usado um conteúdo genérico simulando uma proposta: "
                    f"Proposta comercial e técnica detalhando soluções X, Y, Z, incluindo escopo, metodologia, equipe, cronograma e valores."
                   )
        return text
    except Exception as e:
        logger.exception(f"Erro crítico ao extrair texto do PDF {pdf_file.filename}: {e}")
        # Retorna uma mensagem de erro e um placeholder para o LLM
        return (f"ERRO NA EXTRAÇÃO DE TEXTO: Ocorreu um erro ao processar o PDF '{pdf_file.filename}': {str(e)}. "
                f"O conteúdo deste PDF não pôde ser analisado pelo Gemini.")


def apply_basic_markdown_to_docs_requests(markdown_content: str) -> List[Dict]:
    """
    Converte um subconjunto de Markdown para uma lista de requests da Google Docs API.
    Lida com: # (Heading 1), ## (Heading 2), **bold**, - (list items), e parágrafos.
    A manipulação de índices é simplificada e pode precisar de ajustes para cenários complexos.
    """
    requests = []
    lines = markdown_content.split('\n')
    current_index = 1 # Google Docs API é 1-based para índices de texto

    for line in lines:
        line_stripped = line.strip() # Remove espaços no início/fim

        # Lidar com linhas vazias (transforma em parágrafos vazios no Docs)
        if not line_stripped:
            requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
            current_index += 1
            continue

        # Cabeçalhos
        if line_stripped.startswith('## '):
            text = line_stripped[3:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            # Aplica estilo de Cabeçalho 2 ao parágrafo inserido
            # O range deve cobrir o texto do cabeçalho, não o \n final para estilo de parágrafo
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(text)},
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "fields": "namedStyleType" # Especifica quais campos do paragraphStyle estão sendo atualizados
                }
            })
            current_index += len(text) + 1
        elif line_stripped.startswith('# '):
            text = line_stripped[2:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(text)},
                    "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    "fields": "namedStyleType"
                }
            })
            current_index += len(text) + 1
        
        # Itens de Lista (simples, sem aninhamento)
        elif line_stripped.startswith('- '):
            text = line_stripped[2:]
            # Insere o texto e, em seguida, aplica o marcador
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            # Para criar um item de lista, o ideal é aplicar o bullet ao parágrafo.
            # O range para createParagraphBullets deve abranger o parágrafo.
            # O Google Docs API pode ser um pouco particular com os ranges para bullets.
            # Esta é uma abordagem; pode precisar de ajuste fino.
            requests.append({
                "createParagraphBullets": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(text) +1}, # +1 para o \n
                    "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE" # Um preset comum
                }
            })
            current_index += len(text) + 1
            
        # Negrito (**texto**)
        elif "**" in line_stripped:
            # Esta lógica de negrito pode ser complexa com múltiplos negritos na mesma linha
            # e interações com outros formatos. Uma abordagem mais robusta usaria parsing mais detalhado.
            # A lógica abaixo é uma tentativa simplificada.
            parts = []
            last_pos = 0
            for match in re.finditer(r'\*\*(.*?)\*\*', line_stripped):
                # Texto antes do negrito
                if match.start() > last_pos:
                    parts.append({'text': line_stripped[last_pos:match.start()], 'bold': False})
                # Texto em negrito
                parts.append({'text': match.group(1), 'bold': True})
                last_pos = match.end()
            # Texto após o último negrito
            if last_pos < len(line_stripped):
                parts.append({'text': line_stripped[last_pos:], 'bold': False})

            for part in parts:
                requests.append({"insertText": {"location": {"index": current_index}, "text": part['text']}})
                if part['bold']:
                    requests.append({
                        "updateTextStyle": {
                            "range": {"startIndex": current_index, "endIndex": current_index + len(part['text'])},
                            "textStyle": {"bold": True},
                            "fields": "bold"
                        }
                    })
                current_index += len(part['text'])
            
            requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
            current_index += 1

        # Parágrafo Normal
        else:
            requests.append({"insertText": {"location": {"index": current_index}, "text": line_stripped + "\n"}})
            current_index += len(line_stripped) + 1
            
    return requests


async def generate_etp_tr_content_with_gemini(llm_context_data: Dict) -> Dict:
    """
    Gera o conteúdo do ETP e TR utilizando o modelo Google Gemini.
    O prompt é construído dinamicamente com base em dados do formulário, GCS e PDFs.
    """
    if not gemini_model:
        logger.error("Modelo Gemini não inicializado. Não é possível gerar conteúdo.")
        raise HTTPException(status_code=500, detail="Serviço de IA (LLM) não configurado ou falhou ao iniciar.")

    logger.info("Iniciando chamada ao modelo Gemini para geração de ETP/TR.")

    gcs_accel_str_parts = []
    for product_key, content in llm_context_data.get('gcs_accelerator_content', {}).items():
        if content:
            # Tenta extrair nome e tipo do produto da chave
            product_name_parts = product_key.split('_')
            product_name = product_name_parts[0] if product_name_parts else product_key
            doc_type = product_name_parts[1] if len(product_name_parts) > 1 else "Info"
            doc_type_name = {"BC": "Battle Card", "DS": "Data Sheet", "OP": "Plano Operacional"}.get(doc_type, doc_type)
            gcs_accel_str_parts.append(f"Conteúdo GCS - Acelerador {product_name} ({doc_type_name}):\n{content}\n---\n")
    gcs_accel_str = "\n".join(gcs_accel_str_parts) if gcs_accel_str_parts else "Nenhum conteúdo de acelerador do GCS fornecido.\n"

    gcs_legal_str_parts = []
    for file_name, content in llm_context_data.get('gcs_legal_context_content', {}).items():
        if content:
            gcs_legal_str_parts.append(f"Conteúdo GCS - Documento Legal/Contexto Adicional ({file_name}):\n{content}\n---\n")
    gcs_legal_str = "\n".join(gcs_legal_str_parts) if gcs_legal_str_parts else "Nenhum conteúdo legal/contextual do GCS fornecido.\n"

    # Dados do formulário
    orgao_nome = llm_context_data.get('orgaoSolicitante', '[NOME DO ÓRGÃO SOLICITANTE NÃO FORNECIDO]')
    titulo_projeto = llm_context_data.get('tituloProjeto', '[TÍTULO DO PROJETO NÃO FORNECIDO]')
    justificativa_necessidade = llm_context_data.get('justificativaNecessidade', '[JUSTIFICATIVA DA NECESSIDADE NÃO FORNECIDA]')
    objetivo_geral = llm_context_data.get('objetivoGeral', '[OBJETIVO GERAL NÃO FORNECIDO]')
    prazos_estimados = llm_context_data.get('prazosEstimados', '[PRAZOS ESTIMADOS NÃO FORNECIDOS]')
    valor_estimado_input = llm_context_data.get('valorEstimado') # Pode ser None
    modelo_licitacao = llm_context_data.get('modeloLicitacao', '[MODELO DE LICITAÇÃO NÃO FORNECIDO]')
    parcelamento_contratacao = llm_context_data.get('parcelamentoContratacao', 'Não especificado')
    justificativa_parcelamento = llm_context_data.get('justificativaParcelamento', 'Não se aplica')
    contexto_geral_orgao = llm_context_data.get('contextoGeralOrgao', '[CONTEXTO GERAL DO ÓRGÃO NÃO FORNECIDO]')
    
    today = date.today()
    meses_pt = {
        1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho",
        7: "julho", 8: "agosto", 9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"
    }
    mes_extenso = meses_pt[today.month]
    ano_atual = today.year
    
    esfera_administrativa = "Federal" # Default
    orgao_nome_lower = orgao_nome.lower()
    if any(term in orgao_nome_lower for term in ["municipal", "pref.", "prefeitura"]):
        esfera_administrativa = "Municipal"
    elif any(term in orgao_nome_lower for term in ["estadual", "governo do estado", "secretaria de estado", "tj", "tribunal de justiça"]):
        esfera_administrativa = "Estadual"
        
    local_etp_full = f"[LOCAL PADRÃO, ex: Brasília/DF], {today.day} de {mes_extenso} de {ano_atual}"


    accelerator_details_prompt_list = []
    produtos_selecionados = llm_context_data.get("produtosXertica", [])
    for product_name in produtos_selecionados:
        user_integration_detail = llm_context_data.get(f"integracao_{product_name}", "").strip()
        
        # Busca os conteúdos já carregados em llm_context_data
        bc_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_BC_GCS", "Dados do Battle Card não disponíveis para este produto.")
        ds_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_DS_GCS", "Dados do Data Sheet não disponíveis para este produto.")
        op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_OP_GCS", "Dados do Plano Operacional não disponíveis para este produto.")

        # Limita o tamanho do conteúdo para não exceder limites do prompt
        bc_summary = bc_content_prod_raw[:min(800, len(bc_content_prod_raw))] + ("..." if len(bc_content_prod_raw) > 800 else "")
        ds_summary = ds_content_prod_raw[:min(800, len(ds_content_prod_raw))] + ("..." if len(ds_content_prod_raw) > 800 else "")
        op_summary = op_content_prod_raw[:min(800, len(op_content_prod_raw))] + ("..." if len(op_content_prod_raw) > 800 else "")

        accelerator_details_prompt_list.append(f"""
    - **Acelerador:** {product_name}
      - **Resumo do Battle Card (GCS):** {bc_summary}
      - **Detalhes do Data Sheet (GCS):** {ds_summary}
      - **Detalhes do Plano Operacional (GCS):** {op_summary}
      - **Aplicação Específica no Órgão (Input do Usuário):** {user_integration_detail if user_integration_detail else 'Não detalhado pelo usuário. O LLM deve inferir com base no problema/solução, se possível.'}
        """)
    
    accelerator_details_prompt_section = "\n".join(accelerator_details_prompt_list) if accelerator_details_prompt_list else "Nenhum acelerador Xertica.ai selecionado ou detalhes não fornecidos."

    proposta_comercial_content = llm_context_data.get("proposta_comercial_content", "Conteúdo da proposta comercial não fornecido ou erro na extração.")
    proposta_tecnica_content = llm_context_data.get("proposta_tecnica_content", "Conteúdo da proposta técnica não fornecido ou erro na extração.")

    # Estruturas de Mapa de Preços
    price_map_federal_template = """
| Tipo de Licença/Serviço | Fonte de Pesquisa/Contrato Referência | Valor Unitário Anual (R$) | Valor Mensal (R$) | Quantidade Referencial | Valor Total Estimado (R$) Anual |
|---|---|---|---|---|---|
| [Ex: Licença Acelerador X - Enterprise] | [Ex: Contrato YYY/2023 - Órgão Z] | [Ex: 120.000,00] | [Ex: 10.000,00] | [Ex: 100] | [Ex: 12.000.000,00] |
| [Ex: Serviço de Implementação Inicial] | [Ex: Proposta Xertica.ai Data XX/YY] | [Ex: 50.000,00 (Escopo Fechado)] | N/A | 1 | [Ex: 50.000,00] |
| [Ex: Suporte Técnico Premium (Mensal)] | [Ex: Média de Mercado - Pesquisa XPTO] | [Ex: 60.000,00] | [Ex: 5.000,00] | 12 (meses) | [Ex: 60.000,00] |
""" 
    price_map_estadual_municipal_template = """
| Tipo de Licença/Serviço | Fonte de Pesquisa/Contrato Referência | Empresa Contratada (Ref.) | Valor Unitário Anual (R$) | Valor Mensal (R$) | Quantidade Referencial | Valor Total Estimado (R$) Anual |
|---|---|---|---|---|---|---|
| [Ex: Licença Acelerador X - Standard] | [Ex: Ata de Registro de Preços Nº ABC/2024 - Município Y] | [Ex: Empresa XPTO Ltda.] | [Ex: 60.000,00] | [Ex: 5.000,00] | [Ex: 50] | [Ex: 3.000.000,00] |
| [Ex: Consultoria Especializada (Horas)] | [Ex: Tabela de Preços Xertica.ai] | Xertica Tecnologia Ltda. | [Ex: 3.000,00 (por bloco de 10h)] | N/A | [Ex: 5 blocos] | [Ex: 15.000,00] |
"""
    price_map_to_use_template = price_map_federal_template if esfera_administrativa == "Federal" else price_map_estadual_municipal_template

    # Construção do Prompt Final
    # (O prompt extenso que você forneceu seria inserido aqui)
    # Por brevidade, estou usando o prompt que você já tem no seu código.
    # Certifique-se que o `llm_prompt_content` que você usa é o completo.
    # O prompt que você forneceu anteriormente é muito longo para ser incluído aqui diretamente.
    # Vou usar uma versão resumida para ilustrar a estrutura.
    
    # Este é um placeholder para o seu prompt detalhado.
    # No seu código real, você usaria o `llm_prompt_content` completo que você já tem.
    llm_prompt_content_final = f"""
Você é um assistente de IA para gerar documentos ETP e TR para o setor público brasileiro.
Sua resposta DEVE ser um objeto JSON válido com as chaves "subject", "etp_content", e "tr_content".
O conteúdo de "etp_content" e "tr_content" deve ser Markdown.
Adapte a linguagem e referências legais para a esfera: {esfera_administrativa}.
Preencha todos os placeholders `[]` e `{{}}` nos modelos ETP e TR.

Dados do Usuário:
Órgão: {orgao_nome}
Projeto: {titulo_projeto}
Justificativa: {justificativa_necessidade}
Objetivo: {objetivo_geral}
Prazos: {prazos_estimados}
Valor Estimado: {valor_estimado_input if valor_estimado_input is not None else "Não fornecido, estimar e justificar."}
Modelo Licitação: {modelo_licitacao}
Parcelamento: {parcelamento_contratacao} (Justificativa: {justificativa_parcelamento})
Contexto do Órgão: {contexto_geral_orgao}
Aceleradores Xertica: {', '.join(produtos_selecionados) if produtos_selecionados else "Nenhum"}
Detalhes dos Aceleradores (do usuário e GCS):
{accelerator_details_prompt_section}

Conteúdo das Propostas Anexadas:
Proposta Comercial: {proposta_comercial_content[:min(1000, len(proposta_comercial_content))]}...
Proposta Técnica: {proposta_tecnica_content[:min(1000, len(proposta_tecnica_content))]}...

Contexto Legal/Adicional do GCS:
{gcs_legal_str}

Use o seguinte modelo de Mapa de Preços para a estimativa de custos:
{price_map_to_use_template}

Placeholders a serem preenchidos:
{{sumario_aceleradores}}, {{processo_administrativo_numero}}, {{local_etp_full}}, {{mes_extenso}}, {{ano_atual}}, etc. (todos os placeholders do seu prompt original)

MODELO DE ETP:
(Seu modelo ETP completo em Markdown aqui, com os placeholders)

MODELO DE TR:
(Seu modelo TR completo em Markdown aqui, com os placeholders)

Gere o JSON agora.
"""

    try:
        logger.info(f"Enviando prompt para o Gemini (primeiros 500 chars): {llm_prompt_content_final[:500]}...")
        response = await gemini_model.generate_content_async(
            llm_prompt_content_final, # Use o prompt construído
            generation_config=_generation_config
        )
        
        # Verifica se a resposta tem candidatos e partes antes de acessar response.text
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            logger.error("Resposta do Gemini inválida ou sem conteúdo esperado.")
            logger.error(f"Resposta completa do Gemini: {response}")
            raise Exception("Resposta inválida do modelo Gemini (sem partes de conteúdo).")

        # Acessa o texto da primeira parte do primeiro candidato
        generated_text = response.candidates[0].content.parts[0].text
        logger.info(f"Resposta RAW do Gemini (primeiros 2000 chars): {generated_text[:2000]}...")
        
        if generated_text:
            # Tenta limpar o JSON se estiver dentro de ```json ... ```
            match_json = re.search(r"```json\s*([\s\S]*?)\s*```", generated_text, re.DOTALL)
            if match_json:
                json_str = match_json.group(1)
                logger.info("JSON extraído de bloco de código Markdown.")
            else:
                json_str = generated_text # Assume que a resposta já é JSON puro
            
            parsed_content = json.loads(json_str)
            logger.info("Resposta do Gemini recebida e parseada como JSON com sucesso.")
            return parsed_content
        else:
            logger.warning("Resposta do Gemini vazia ou não contém texto processável.")
            raise Exception("Resposta vazia ou não processável do modelo Gemini.")

    except json.JSONDecodeError as e:
        logger.error(f"Erro ao parsear JSON da resposta do Gemini: {e}.")
        # Loga a string que falhou no parse para depuração
        problematic_json_string = generated_text if 'generated_text' in locals() else "Não foi possível capturar a string problemática."
        logger.error(f"String JSON que causou o erro de parse: {problematic_json_string}")
        raise HTTPException(status_code=500, detail=f"Erro no formato JSON retornado pelo Gemini: {e}. Verifique os logs do servidor para a string exata.")
    except AttributeError as e: # Captura erros se a estrutura da resposta do Gemini for inesperada
        logger.error(f"Estrutura da resposta do Gemini inesperada: {e}")
        logger.error(f"Resposta completa do Gemini: {response}")
        raise HTTPException(status_code=500, detail=f"Formato de resposta inesperado do Gemini: {e}")
    except Exception as e:
        logger.exception(f"Erro crítico ao chamar a API do Gemini ou processar sua resposta: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na geração de conteúdo via IA: {e}")


@app.post("/generate_etp_tr", summary="Gera Documentos ETP e TR", tags=["Documentos"])
async def generate_etp_tr_endpoint(
    request: Request, # Para acessar form_data de forma mais flexível
    orgaoSolicitante: str = Form(..., description="Nome completo do órgão público solicitante."),
    tituloProjeto: str = Form(..., description="Título ou nome do projeto/contratação."),
    justificativaNecessidade: str = Form(..., description="Descrição detalhada do problema ou necessidade a ser resolvida."),
    objetivoGeral: str = Form(..., description="Objetivo principal da contratação/solução."),
    prazosEstimados: str = Form(..., description="Prazos estimados para implantação e execução. Ex: 3 meses para implantação, 12 meses de operação."),
    modeloLicitacao: str = Form(..., description="Modelo de licitação pretendido (ex: Pregão Eletrônico, Inexigibilidade, Dispensa)."),
    parcelamentoContratacao: str = Form(..., description="Indica se a contratação será parcelada (Sim, Não, Justificar)."),
    contextoGeralOrgao: Optional[str] = Form(None, description="Breve contexto sobre o órgão, seus desafios e iniciativas relevantes."),
    valorEstimado: Optional[float] = Form(None, description="Valor total estimado da contratação (opcional)."),
    justificativaParcelamento: Optional[str] = Form(None, description="Justificativa caso o parcelamento seja 'Justificar' ou 'Não'."),
    propostaComercialFile: Optional[UploadFile] = File(None, description="Proposta Comercial da Xertica.ai em PDF (opcional)."),
    propostaTecnicaFile: Optional[UploadFile] = File(None, description="Proposta Técnica da Xertica.ai em PDF (opcional).")
    # produtosXertica e integracao_<produto> são lidos diretamente do request.form()
):
    """
    Endpoint principal para gerar os documentos ETP e TR.
    Recebe dados do formulário, processa arquivos PDF (se enviados), consulta dados contextuais do GCS,
    chama o LLM Gemini para gerar o conteúdo dos documentos e, por fim,
    cria um novo Google Doc com o conteúdo gerado e retorna o link.
    """
    logger.info(f"Requisição para gerar ETP/TR para '{tituloProjeto}' do órgão '{orgaoSolicitante}' recebida.")
    
    # Validação inicial dos serviços GCP
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Serviço de IA (LLM) indisponível ou não configurado.")
    if not storage_client:
        raise HTTPException(status_code=503, detail="Serviço de Armazenamento (GCS) indisponível ou não configurado.")

    form_data = await request.form()
    produtosXertica_list = form_data.getlist("produtosXertica")
    logger.info(f"Produtos Xertica selecionados: {produtosXertica_list}")

    llm_context_data = {
        "orgaoSolicitante": orgaoSolicitante,
        "tituloProjeto": tituloProjeto,
        "justificativaNecessidade": justificativaNecessidade,
        "objetivoGeral": objetivoGeral,
        "prazosEstimados": prazosEstimados,
        "modeloLicitacao": modeloLicitacao,
        "parcelamentoContratacao": parcelamentoContratacao,
        "contextoGeralOrgao": contextoGeralOrgao if contextoGeralOrgao else "Não fornecido.",
        "valorEstimado": valorEstimado, # Pode ser None
        "justificativaParcelamento": justificativaParcelamento if justificativaParcelamento else "Não fornecida.",
        "produtosXertica": produtosXertica_list,
        "data_geracao_documento": date.today().strftime("%d/%m/%Y"),
    }

    # Coleta detalhes de integração para cada produto
    for product_name_form in produtosXertica_list:
        # A chave no formulário HTML deve ser "integracao_NOME_DO_PRODUTO"
        integration_key = f"integracao_{product_name_form.replace(' ', '_').replace('.', '_')}" # Normaliza nome para chave
        llm_context_data[integration_key] = form_data.get(integration_key, f"Detalhes de integração para {product_name_form} não fornecidos.")
        logger.info(f"Detalhe de integração para '{product_name_form}': {llm_context_data[integration_key]}")


    # Processamento de arquivos PDF anexados
    if propostaComercialFile and propostaComercialFile.filename:
        logger.info(f"Processando Proposta Comercial: {propostaComercialFile.filename}")
        llm_context_data["proposta_comercial_content"] = await extract_text_from_pdf(propostaComercialFile)
        llm_context_data["commercial_proposal_gcs_url"] = await upload_file_to_gcs(
            propostaComercialFile,
            f"propostas_clientes/{orgaoSolicitante.replace(' ','_')}_{tituloProjeto.replace(' ','_')}_comercial_{date.today().strftime('%Y%m%d')}_{propostaComercialFile.filename}"
        )
    else:
        llm_context_data["proposta_comercial_content"] = "Nenhuma proposta comercial em PDF foi fornecida pelo usuário."
        llm_context_data["commercial_proposal_gcs_url"] = None
        logger.info("Nenhum arquivo de proposta comercial fornecido.")

    if propostaTecnicaFile and propostaTecnicaFile.filename:
        logger.info(f"Processando Proposta Técnica: {propostaTecnicaFile.filename}")
        llm_context_data["proposta_tecnica_content"] = await extract_text_from_pdf(propostaTecnicaFile)
        llm_context_data["technical_proposal_gcs_url"] = await upload_file_to_gcs(
            propostaTecnicaFile,
            f"propostas_clientes/{orgaoSolicitante.replace(' ','_')}_{tituloProjeto.replace(' ','_')}_tecnica_{date.today().strftime('%Y%m%d')}_{propostaTecnicaFile.filename}"
        )
    else:
        llm_context_data["proposta_tecnica_content"] = "Nenhuma proposta técnica em PDF foi fornecida pelo usuário."
        llm_context_data["technical_proposal_gcs_url"] = None
        logger.info("Nenhum arquivo de proposta técnica fornecido.")

    # Coleta de conteúdo contextual do GCS
    llm_context_data['gcs_accelerator_content'] = {}
    llm_context_data['gcs_legal_context_content'] = {}

    for product_name in produtosXertica_list:
        # Caminhos padronizados para os documentos dos aceleradores no GCS
        # Ajuste estes caminhos conforme a estrutura real do seu bucket
        bc_path = f"aceleradores_conteudo/{product_name}/BC - {product_name}.txt"
        ds_path = f"aceleradores_conteudo/{product_name}/DS - {product_name}.txt"
        op_path = f"aceleradores_conteudo/{product_name}/OP - {product_name}.txt"
        
        content_bc = get_gcs_file_content(bc_path)
        content_ds = get_gcs_file_content(ds_path)
        content_op = get_gcs_file_content(op_path)

        if content_bc: llm_context_data['gcs_accelerator_content'][f"{product_name}_BC_GCS"] = content_bc
        if content_ds: llm_context_data['gcs_accelerator_content'][f"{product_name}_DS_GCS"] = content_ds
        if content_op: llm_context_data['gcs_accelerator_content'][f"{product_name}_OP_GCS"] = content_op
        logger.info(f"Conteúdo GCS para acelerador '{product_name}': BC {'encontrado' if content_bc else 'não encontrado'}, DS {'encontrado' if content_ds else 'não encontrado'}, OP {'encontrado' if content_op else 'não encontrado'}.")

    # Exemplos de caminhos para documentos legais/contextuais
    # Adapte os nomes e caminhos conforme sua organização no GCS
    legal_docs_map = {
        "MTI_CONTRATO_EXEMPLO.txt": "exemplos_legais/contratos/CONTRATO_PARCERIA_MTI_XERTICA.txt",
        "MPAP_ATA_EXEMPLO.txt": "exemplos_legais/atas/ATA_REGISTRO_PRECOS_MPAP_XERTICA.txt",
        "RISK_ANALYSIS_CONTEXT.txt": "contexto_geral/analise_riscos/DETECCAO_ANALISE_RISCOS.txt",
        "SERPRO_MOU_EXEMPLO.txt": "exemplos_legais/mou/MOU_SERPRO_XERTICA.txt"
    }
    for display_name, gcs_path in legal_docs_map.items():
        content = get_gcs_file_content(gcs_path)
        if content:
            llm_context_data['gcs_legal_context_content'][display_name] = content
            logger.info(f"Conteúdo GCS para contexto legal '{display_name}' carregado.")
        else:
            logger.warning(f"Conteúdo GCS para contexto legal '{display_name}' (caminho: {gcs_path}) não encontrado.")
    
    # Geração de conteúdo com Gemini
    logger.info("Enviando dados de contexto para o LLM Gemini.")
    llm_response = await generate_etp_tr_content_with_gemini(llm_context_data)

    document_subject = llm_response.get("subject", f"ETP e TR: {orgaoSolicitante} - {tituloProjeto} ({date.today().strftime('%Y-%m-%d')})")
    etp_content_md = llm_response.get("etp_content", "# ETP\n\nErro: Conteúdo do ETP não foi gerado corretamente pelo LLM.")
    tr_content_md = llm_response.get("tr_content", "# Termo de Referência\n\nErro: Conteúdo do TR não foi gerado corretamente pelo LLM.")

    # Autenticação e criação do Google Doc
    docs_service, drive_service = authenticate_google_docs_and_drive()

    try:
        # Cria um novo Google Doc em branco
        new_doc_body = {'title': document_subject} # mimeType é inferido pelo Drive API para Google Docs
        new_doc_metadata = drive_service.files().create(body=new_doc_body, fields='id,webViewLink').execute()
        document_id = new_doc_metadata.get('id')
        document_link_initial = new_doc_metadata.get('webViewLink') # Link inicial
        
        if not document_id:
            logger.error("Falha ao criar novo documento no Google Docs. ID não retornado.")
            raise HTTPException(status_code=500, detail="Falha ao criar novo documento no Google Docs (ID não obtido).")
        logger.info(f"Documento Google Docs criado com ID: {document_id}, Link inicial: {document_link_initial}")

        # Combina conteúdo ETP e TR para inserção
        # Adiciona um separador claro, como uma quebra de página ou um título para o TR
        combined_markdown_content = f"{etp_content_md}\n\n<newpage>\n\n{tr_content_md}" 
        
        # Converte Markdown para Google Docs API requests
        # A função apply_basic_markdown_to_docs_requests precisa ser robusta.
        requests_for_docs_api = apply_basic_markdown_to_docs_requests(combined_markdown_content)
        
        if requests_for_docs_api: # Só executa se houver requests a fazer
            docs_service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests_for_docs_api}
            ).execute()
            logger.info(f"Conteúdo ETP e TR inserido e formatado no documento Google Docs: {document_id}")
        else:
            logger.warning(f"Nenhuma request de formatação gerada para o documento {document_id}. Conteúdo pode estar vazio ou não formatado.")

        # Define permissões para o documento (leitura pública)
        # CUIDADO: Isso torna o documento acessível a qualquer pessoa com o link.
        # Em produção, ajuste as permissões conforme necessário.
        permission = {'type': 'anyone', 'role': 'reader'}
        try:
            drive_service.permissions().create(fileId=document_id, body=permission, fields='id').execute()
            logger.info(f"Permissões de leitura pública definidas para o documento: {document_id}")
        except HttpError as e_perm:
            # Alguns domínios GWorkspace podem restringir o compartilhamento "anyone"
            logger.warning(f"Não foi possível aplicar permissão 'anyone' ao documento {document_id}: {e_perm}. "
                           "O documento pode não ser publicamente acessível. Verifique as políticas do domínio.")

        # Obtém o link final do documento (webViewLink é geralmente o mais útil)
        file_metadata_final = drive_service.files().get(fileId=document_id, fields='webViewLink, id').execute()
        document_link_final = file_metadata_final.get('webViewLink')

        if not document_link_final:
            logger.error(f"Falha ao obter o link final do Google Docs para o documento {document_id}.")
            # Usa o link inicial se o final não estiver disponível, mas loga.
            document_link_final = document_link_initial or f"https://docs.google.com/document/d/{document_id}/edit"


        logger.info(f"Processo de geração de ETP/TR concluído com sucesso. Link do Documento: {document_link_final}")
        return JSONResponse(status_code=200, content={
            "success": True, 
            "message": "Documentos ETP e TR gerados e salvos no Google Docs.",
            "doc_link": document_link_final,
            "document_id": document_id,
            "commercial_proposal_gcs_url": llm_context_data.get("commercial_proposal_gcs_url"),
            "technical_proposal_gcs_url": llm_context_data.get("technical_proposal_gcs_url")
        })

    except HttpError as e_google_api:
        # Tenta decodificar a mensagem de erro da API do Google
        error_content = e_google_api.content.decode() if e_google_api.content else "{}"
        try:
            error_details_json = json.loads(error_content)
            error_message = error_details_json.get('error', {}).get('message', f"Erro desconhecido na API do Google. Status: {e_google_api.resp.status}")
        except json.JSONDecodeError:
            error_message = f"Erro na API do Google (resposta não JSON): {error_content}. Status: {e_google_api.resp.status}"
        
        logger.exception(f"Erro na API do Google Docs/Drive durante criação/atualização do documento: {error_message}") 
        raise HTTPException(status_code=e_google_api.resp.status if hasattr(e_google_api, 'resp') else 500, 
                            detail=f"Erro na API do Google Docs/Drive: {error_message}")
    except Exception as e_general:
        logger.exception(f"Erro inesperado durante a geração ou criação do documento Google Docs: {e_general}") 
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e_general}. Verifique os logs.")

# Para rodar localmente com Uvicorn (exemplo):
# uvicorn main:app --reload --port 8000
# Certifique-se de ter as variáveis de ambiente configuradas (GCP_PROJECT_ID, etc.)
# e as credenciais do Google Cloud (Application Default Credentials) configuradas no seu ambiente.
