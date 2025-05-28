"""
Projeto: Gerador de ETP e Termo de Referência com IA para Documentos Xertica.ai
Módulo: main.py
Descrição: Backend da aplicação FastAPI que orquestra a geração de documentos ETP (Estudo Técnico Preliminar)
           e TR (Termo de Referência) utilizando o modelo Gemini 1.5 Flash do Google Vertex AI.
           Integra-se com Google Cloud Storage para armazenamento de anexos e Google Docs API para
           criação e formatação dos documentos finais.
Autor: Xertica.ai - Assistente de IA
Data: 2024-07-23
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

from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pypdf import PdfReader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


load_dotenv()


app = FastAPI(
    title="Gerador de ETP e TR Xertica.ai",
    description="Backend inteligente para gerar documentos ETP e TR com IA da Xertica.ai.",
    version="0.1.0"
)


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.getenv("GCP_PROJECT_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "docsorgaospublicos")


if not GCP_PROJECT_ID:
    logger.error("GCP_PROJECT_ID não está configurado.")
    raise Exception("GCP_PROJECT_ID não configurado. Por favor, defina a variável de ambiente.")

if not GCP_PROJECT_LOCATION:
    logger.error("GCP_PROJECT_LOCATION não está configurado.")
    raise Exception("GCP_PROJECT_LOCATION não configurado. Por favor, defina a variável de ambiente.")


try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_PROJECT_LOCATION)
    gemini_model = GenerativeModel("gemini-1.5-flash-001")
    _generation_config = GenerationConfig(temperature=0.7, max_output_tokens=8192, response_mime_type="application/json")
    logger.info(f"Vertex AI inicializado com projeto '{GCP_PROJECT_ID}' e localização '{GCP_PROJECT_LOCATION}'. Modelo Gemini-1.5-flash-001 carregado.")
except Exception as e:
    logger.exception(f"Erro ao inicializar Vertex AI ou carregar modelo Gemini: {e}")
    raise HTTPException(status_code=500, detail=f"Falha ao iniciar LLM: {e}. Verifique as permissões da Service Account e as variáveis GCP_PROJECT_ID/GCP_PROJECT_LOCATION.")


try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Google Cloud Storage client inicializado.")
except Exception as e:
    logger.exception(f"Erro ao inicializar Google Cloud Storage client: {e}")
    raise HTTPException(status_code=500, detail=f"Falha ao inicializar GCS: {e}")


def authenticate_google_docs_and_drive():
    """Autentica com as APIs do Google Docs e Drive."""
    try:
        docs_service = build('docs', 'v1', cache_discovery=False)
        drive_service = build('drive', 'v3', cache_discovery=False)
        logger.info("Google Docs e Drive services inicializados.")
        return docs_service, drive_service
    except Exception as e:
        logger.exception(f"Erro ao autenticar/inicializar Google Docs/Drive APIs: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na autenticação da API do Google Docs/Drive: {e}. Verifique as permissões da Service Account (Docs API Editor, Drive API File Creator).")

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
        await upload_file.seek(0) 

        blob.upload_from_string(contents, content_type=upload_file.content_type)
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
            logger.warning(f"O texto extraído de {pdf_file.filename} está vazio ou contém apenas espaços em branco. Isso pode indicar um PDF baseado em imagem (escaneado) ou com um layout muito complexo para pypdf.")
            return (f"**ATENÇÃO:** Não foi possível extrair texto legível do PDF '{pdf_file.filename}'. "
                    f"O arquivo pode ser um PDF de imagem (escaneado) ou ter um formato que dificulta a extração de texto por pypdf. "
                    f"Para análise pelo Gemini, será usado um conteúdo genérico simulando uma proposta comercial/técnica: "
                    f"A presente proposta detalha as funcionalidades da solução da Xertica.ai, incluindo escopo dos serviços, "
                    f"metodologia de implementação, requisitos técnicos, cronograma estimado e valores associados.")
        return text
    except Exception as e:
        logger.exception(f"Erro crítico ao extrair texto do PDF {pdf_file.filename}: {e}")
        return (f"**ERRO NA EXTRAÇÃO DE TEXTO:** Ocorreu um erro ao processar o PDF '{pdf_file.filename}': {str(e)}. "
                f"O conteúdo deste PDF não pôde ser analisado pelo Gemini. "
                f"Considere refazer o upload ou fornecer as informações de forma manual.")


def apply_basic_markdown_to_docs_requests(markdown_content: str) -> List[Dict]:
    """
    Converte um subconjunto de Markdown para uma lista de requests da Google Docs API.
    Lida com: # (Heading 1), ## (Heading 2), **bold**, - (list items), e parágrafos.
    A manipulação de índices é simplificada e pode precisar de ajustes para cenários complexos.
    Um `requests` é adicionado para `insertPageBreak` quando detecta `<NEWPAGE>`.
    """
    requests = []
    lines = markdown_content.split('\n')
    current_index = 1

    for line in lines:
        line_stripped = line.strip()

        if line_stripped == "<NEWPAGE>":
            requests.append({"insertPageBreak": {"location": {"index": current_index}}})
            current_index += 1
            continue

        if not line_stripped:
            requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
            current_index += 1
            continue

        if line_stripped.startswith('## '):
            text = line_stripped[3:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(text)},
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "fields": "namedStyleType"
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
        
        elif line_stripped.startswith('* ') or line_stripped.startswith('- '):
            text = line_stripped[2:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            requests.append({
                "createParagraphBullets": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(text) +1},
                    "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
                }
            })
            current_index += len(text) + 1
            
        elif "**" in line_stripped:
            parts = []
            last_pos = 0
            for match in re.finditer(r'\*\*(.*?)\*\*', line_stripped):
                if match.start() > last_pos:
                    parts.append({'text': line_stripped[last_pos:match.start()], 'bold': False})
                parts.append({'text': match.group(1), 'bold': True})
                last_pos = match.end()
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
    # Loop para construir a string de conteúdo de aceleradores GCS
    for product_key_in_dict, content in llm_context_data.get('gcs_accelerator_content', {}).items():
        if content:
            # Extrai o tipo de documento (BC, DS, OP, OP_GCP, etc.) do final da chave
            doc_type_match = re.search(r'_(BC|DS|OP|OP_GCP|OP_GMP|OP_GWS)$', product_key_in_dict)
            doc_type_indicator = doc_type_match.group(1) if doc_type_match else "Unknown"
            
            # Extrai o nome do produto (removendo o sufixo do tipo de documento)
            product_name_actual = product_key_in_dict[:-len(doc_type_indicator)-1] if doc_type_match else product_key_in_dict

            doc_type_name = {
                "BC": "Battle Card", 
                "DS": "Data Sheet", 
                "OP": "Plano Operacional",
                "OP_GCP": "Plano Operacional (Acel. GCP)",
                "OP_GMP": "Plano Operacional (Acel. GMP)",
                "OP_GWS": "Plano Operacional (Acel. GWS)",
            }.get(doc_type_indicator, doc_type_indicator)
            
            gcs_accel_str_parts.append(f"Conteúdo GCS - Acelerador {product_name_actual} ({doc_type_name}):\n{content}\n---\n")
    gcs_accel_str = "\n".join(gcs_accel_str_parts) if gcs_accel_str_parts else "Nenhum conteúdo de acelerador do GCS fornecido.\n"

    gcs_legal_str_parts = []
    for file_name, content in llm_context_data.get('gcs_legal_context_content', {}).items():
        if content:
            gcs_legal_str_parts.append(f"Conteúdo GCS - Documento Legal/Contexto Adicional ({file_name}):\n{content}\n---\n")
    gcs_legal_str = "\n".join(gcs_legal_str_parts) if gcs_legal_str_parts else "Nenhum conteúdo legal/contextual do GCS fornecido.\n"

    # Prepara variáveis Python para injeção no prompt (para evitar f-string aninhadas)
    orgao_nome_py = llm_context_data.get('orgaoSolicitante', 'o Órgão Solicitante')
    titulo_projeto_py = llm_context_data.get('tituloProjeto', 'uma iniciativa')
    justificativa_necessidade_py = llm_context_data.get('justificativaNecessidade', 'um problema genérico.')
    objetivo_geral_py = llm_context_data.get('objetivoGeral', 'um objetivo ambicioso.')
    prazos_estimados_py = llm_context_data.get('prazosEstimados', 'prazos a serem definidos.')
    valor_estimado_input_py = llm_context_data.get('valorEstimado')
    modelo_licitacao_py = llm_context_data.get('modeloLicitacao', 'uma modalidade padrão.')
    parcelamento_contratacao_py = llm_context_data.get('parcelamentoContratacao', 'Não especificado.')
    justificativa_parcelamento_py = llm_context_data.get('justificativaParcelamento', 'Não se aplica.')
    contexto_geral_orgao_py = llm_context_data.get('contextoGeralOrgao', '')
    
    today = date.today()
    meses_pt = {
        1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho",
        7: "julho", 8: "agosto", 9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"
    }
    mes_extenso_py = meses_pt[today.month]
    ano_actual_py = today.year
    today_formatted = today.strftime('%d/%m/%Y')
    
    esfera_administrativa_py = "Federal"
    orgao_nome_lower = orgao_nome_py.lower()
    if any(term in orgao_nome_lower for term in ["municipal", "pref.", "prefeitura"]):
        esfera_administrativa_py = "Municipal"
    elif any(term in orgao_nome_lower for term in ["estadual", "governo do estado", "secretaria de estado", "tj", "tribunal de justiça"]):
        esfera_administrativa_py = "Estadual"
        
    local_etp_full_py = f"[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO, ex: Brasília/DF], {today.day} de {mes_extenso_py} de {ano_actual_py}"

    # Prepara o JSON para o prompt
    llm_context_data_json = json.dumps(llm_context_data, indent=2, ensure_ascii=False)

    # Prepara os sumários dos aceleradores para o prompt
    accelerator_details_prompt_list = []
    produtos_selecionados = llm_context_data.get("produtosXertica", []) # Isso já é a lista de nomes originais
    for product_name in produtos_selecionados: # product_name aqui é o nome original, ex: "Meet Transcriber"
        # Access user integration detail using the normalized key stored earlier
        integration_key_in_llm_context = f"integracao_{product_name.replace(' ', '_').replace('.', '_')}"
        user_integration_detail = llm_context_data.get(integration_key_in_llm_context, "").strip()
        
        # Acessar o conteúdo do GCS usando a chave que está no llm_context_data, que usa o nome original
        bc_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_BC", "")
        ds_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_DS", "")
        
        op_content_prod_raw = ""
        # Lógica para OP com nomes especiais GWS/GCP/GMP precisa ser robusta aqui
        if product_name == "GCP": op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"GCP_OP_GCP", "")
        elif product_name == "GMP": op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"GMP_OP_GMP", "")
        elif product_name == "GWS": op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"GWS_OP_GWS", "")
        else: op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_OP", "")

        bc_summary = bc_content_prod_raw[:min(1000, len(bc_content_prod_raw))] + ("..." if len(bc_content_prod_raw) > 1000 else "") if bc_content_prod_raw else "Dados do Battle Card não disponíveis para este produto."
        ds_summary = ds_content_prod_raw[:min(1000, len(ds_content_prod_raw))] + ("..." if len(ds_content_prod_raw) > 1000 else "") if ds_content_prod_raw else "Dados do Data Sheet não disponíveis para este produto."
        op_summary = op_content_prod_raw[:min(1000, len(op_content_prod_raw))] + ("..." if len(op_content_prod_raw) > 1000 else "") if op_content_prod_raw else "Dados do Plano Operacional não disponíveis para este produto."


        accelerator_details_prompt_list.append(f"""
    - **Acelerador:** {product_name}
      - **Resumo do Battle Card (GCS):** {bc_summary}
      - **Detalhes do Data Sheet (GCS):** {ds_summary}
      - **Detalhes do Plano Operacional (GCS):** {op_summary}
      - **Aplicação Específica no Órgão (Input do Usuário):** {user_integration_detail if user_integration_detail else 'Nenhum detalhe de integração fornecido. O LLM deve inferir a aplicação com base no problema/solução, nos documentos do acelerador e no contexto Xertica.ai'}
        """)
    
    gcs_accel_str_for_prompt = "\n".join(accelerator_details_prompt_list) if accelerator_details_prompt_list else "Nenhum acelerador Xertica.ai selecionado ou detalhes não fornecidos."


    proposta_comercial_content_py = llm_context_data.get("proposta_comercial_content", "Conteúdo da proposta comercial não fornecido ou erro na extração.")
    proposta_tecnica_content_py = llm_context_data.get("proposta_tecnica_content", "Conteúdo da proposta técnica não fornecido ou erro na extração.")

    # Formata valor_estimado_input para o prompt
    valor_estimado_input_final = (f"{valor_estimado_input:.2f}".replace('.', ',') if valor_estimado_input is not None else '[[VALOR ESTIMADO PELO LLM COM BASE NAS INFORMAÇÕES FORNECIDAS]]')

    # Formata justificativa_parcelamento para o prompt
    justificativa_parcelamento_final_py = (justificativa_parcelamento_py if parcelamento_contratacao_py == 'Justificar' and justificativa_parcelamento_py else f"A decisão por {('parcelar' if parcelamento_contratacao_py == 'Sim' else 'não parcelar')} a contratação foi embasada na busca por {('maior flexibilidade na gestão e adaptação do projeto por fases, permitindo entregas incrementais e avaliação contínua dos resultados. Isso mitiga riscos e permite o aprimoramento progressivo da solução, além de otimizar a gestão orçamentária.' if parcelamento_contratacao_py == 'Sim' else 'garantir a integralidade da solução e a sinergia entre seus componentes, otimizando o processo de implementação e a entrega de resultados completos. O não parcelamento minimiza a fragmentação de responsabilidades, assegura a interoperabilidade plena e acelera o tempo de valor da solução para o órgão.')}.")

    # Nomes dos produtos selecionados para o prompt
    nomes_produtos_selecionados_py = ', '.join(produtos_selecionados) if produtos_selecionados else '[LISTAR ACELERADORES SELECIONADOS]'


    # O corpo do prompt é uma string literal, formatada no final com .format()
    llm_prompt_content_template = """
Você é um assistente de IA altamente especializado em elaboração de documentos técnicos e legais para o setor público brasileiro (esferas Federal, Estadual e Municipal), com expertise em licitações (Lei nº 14.133/2021, Lei 13.303/2016 e outras regulamentações específicas, como o Decreto Estadual 21.872/2023 se aplicável), e nas soluções de Inteligência Artificial da Xertica.ai.

Sua tarefa é gerar duas seções completas (ETP e TR) em Markdown.
O conteúdo gerado DEVE ser em PROSA rica e detalhada, com análises aprofundadas, justificativas robustas e descrições técnicas claras, não apenas listas de bullet points. Para listas que são naturalmente itens (ex: requisitos, obrigações), use o formato de lista Markdown (`*` ou `-`).
Siga rigorosamente os modelos de estrutura ETP e TR (que serão fornecidos abaixo) e adapte todo o conteúdo à esfera administrativa do órgão solicitante ({esfera_administrativa_py}).
Você deve preencher todos os `[ ]` e `{{}}` nos modelos com informações contextualmente relevantes e gerar todo o texto dinâmico e analítico necessário.

Sua resposta FINAL DEVE ser UM OBJETO JSON VÁLIDO.

Formato do JSON de Saída:
```json
{{
   "subject": "Título Descritivo do Documento (ETP e TR)",
   "etp_content": "Conteúdo COMPLETO do ETP em Markdown, preenchendo todos os `[]` e `{{}}` e gerando texto necessário, com prosa detalhada.",
   "tr_content": "Conteúdo COMPLETO do TR em Markdown, preenchendo todos os `[]` e `{{}}` e gerando texto necessário, com prosa detalhada."
}}
Regras Detalhadas para Geração de Conteúdo:

Adaptação Linguística e Legal Específica por Esfera: Adapte toda a linguagem e referências legais (por exemplo, "Lei Orçamentária Anual da União" vs. "Lei Orçamentária Anual do Estado/Município", "MINISTÉRIO PÚBLICO da {{{{NÍVEL ADMINISTRATIVO}}}}", ou legislações específicas de estado/município quando relevante) com base na esfera administrativa {esfera_administrativa_py} do {orgao_nome_py}.
Racionalize Informações Incompletas e Preencha Placeholders: Para campos que "deverão ser preenchidos pelo órgão" e que não estão na entrada (como número de processo, nomes de responsáveis técnicos, dados orçamentários), o LLM deve: Preencher com um placeholder claro em negrito e entre [ ] como [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO].
No caso de número de processo (Processo Administrativo nº), gerar um formato genérico e consistente (XXXXXX/{{{{ANO_ATUAL_LLM}}}}).
Para CNAE, sugira um CNAE de TI comum (ex: "6204-0/00 - Consultoria em tecnologia da informação") e ADICIONE TEXTO explicando que "o CNAE específico deverá ser confirmado e preenchido pelo órgão".
Para prazos de recebimento e faturamento/pagamento, utilize valores razoáveis e comuns para contratos públicos (ex: "5 (cinco) dias úteis", "15 (quinze) dias corridos").
Preencha os valores da tabela de Mapa de Preços de Referência com estimativas realistas, explicando que o valor está em consonância com dados de mercado e propostas anteriores. Se o {valor_estimado_input_final} global foi fornecido, a soma da tabela deve se aproximar dele, e os valores unitários e mensais devem ser coerentes.
Análise Profunda e Detalhamento (Deep Research / RAG): Utilize rigorosamente as informações dos Battle Cards, Data Sheets e Operational Plans (do GCS) e o conteúdo extraído das propostas anexadas (PDFs) para realizar uma "deep research" e descrever as funcionalidades dos aceleradores da Xertica.ai.
CRÍTICO: Nas seções de "Descrição da Solução", "Requisitos", "Levantamento de Mercado" e "Justificativa da Solução", o LLM deve detalhar o que a Xertica.ai faz de ÚNICO ou MELHOR (ex: integração nativa com GCP, especialização no setor público, notória especialização, vivência em projetos similares, adaptabilidade, inovação, etc.) em comparação com alternativas, usando o contexto do GCS e PDFs.
Descreva como as soluções Xertica.ai diretamente resolvem a {justificativa_necessidade_py} e atingem o {objetivo_geral_py} do órgão, detalhando os benefícios e impactos esperados em prosa analítica.
Em "Levantamento de Mercado", além de justificar a Xertica.ai, pode-se brevemente mencionar tipos de soluções alternativas e por que a Xertica.ai é a mais vantajosa (ex: inovação superior, integração facilitada, foco no setor público).
Resultados Esperados: Detalhe os resultados esperados com indicadores qualitativos e, se possível, quantitativos (ex: "redução de até X% no tempo de atendimento", "aumento de Y% na satisfação do usuário").
Justificativa Legal Robusta: Para o {modelo_licitacao_py}, use o conhecimento da Lei 14.133/2021 e da Lei 13.303/2016 e o contexto legal do GCS (MTI, MPAP, SERPRO MoU, ABES, Riscos) para fornecer justificativas legais detalhadas, citando artigos relevantes e explicando a aplicabilidade.
Formatação Markdown: A saída DEVE ser Markdown válido. Use # para H1, ## para H2, ### para H3, * ou - para listas NÃO aninhadas. Garanta que as tabelas sejam formatadas corretamente.
DADOS FORNECIDOS PELO USUÁRIO (Órgão Solicitante):

{llm_context_data_json}
CONTEÚDO EXTRAÍDO DAS PROPOSTAS XERTICA.AI (Anexos PDF):
Proposta Comercial: {proposta_comercial_content_py}
Proposta Técnica: {proposta_tecnica_content_py}

MAPA DE PREÇOS DE REFERÊNCIA PARA CONTRATAÇÃO (Estrutura Fornecida para Orientação do LLM):
Utilize esta estrutura para fundamentar a seção de estimativa de preço. O LLM DEVE PREENCHER OS VALORES de forma realista e justificada, mesmo que o input valorEstimado não seja fornecido. Se valorEstimado for fornecido, a soma da tabela deve se aproximar dele.
{price_map_to_use_template}

CONTEÚDO DE CONTEXTO GCS (Battle Cards, Data Sheets, OP, Documentos Legais):
{gcs_accel_str}
{gcs_legal_str}

Mapeamento de Placeholders para Preenchimento (Para o LLM):

{{sumario_aceleradores}}: Resumo dos aceleradores selecionados.
{{processo_administrativo_numero}}: Processo administrativo (genérico ou do contexto, ex: XXXXXX/{{{{ANO_ATUAL_LLM}}}}).
{{local_etp_full}}: Onde local_etp_full foi definido ("{local_etp_full_py}").
{{mes_extenso}}: Mês atual por extenso ({mes_extenso_py}).
{{ano_atual}}: Ano atual ({ano_actual_py}).
{{introducao_etp}}: Detalha o texto de introdução do ETP.
{{referencia_in_sgd_me}}: Adiciona a referência legal da IN SGD/ME nº 94/2022.
{{problema_necessidade}}: Desenvolve {justificativa_necessidade_py}.
{{necessidades_negocio}}: Desenvolve necessidades de negócio com base em {justificativa_necessidade_py} e {objetivo_geral_py}.
{{requisitos_tecnicos_funcionais}}: Desenvolve requisitos técnicos e funcionais detalhados, baseados nos aceleradores e na proposta_tecnica_content.
{{levantamento_mercado}}: Analisa mercado, diferenciais da Xertica.ai usando gcs_accelerator_content, proposta_tecnica_content_py, e exemplos como PROCERGS MPRS TJES.
{{estimativa_demanda}}: Estima demanda com base na {justificativa_necessidade_py} e {objetivo_geral_py}.
{{mapa_comparativo_custos}}: Tabela e texto justificando a estimativa de custos (LLM DEVE PREENCHER A TABELA COM VALORES).
{{valor_estimado_total_etp}}: Valor final e justificativa. Se {valor_estimado_input_py} foi fornecido ({valor_estimado_input_final}), usar como base. Senão, o LLM estima.
{{descricao_solucao_etp}}: Descrição da solução, baseada nos aceleradores e proposta_tecnica_content.
{{parcelamento_justificativa}}: Justificativa para parcelamento ({parcelamento_contratacao_py}, {justificativa_parcelamento_final_py}).
{{providencias_tomadas}}: Lista de providências (genéricas).
{{declaracao_viabilidade}}: Declaração de viabilidade.
{{nomes_cargos_responsaveis}}: Nomes e cargos de responsáveis.
{{local_data_aprovacao}}: Local e data para aprovação.
{{cidade_uf_tr}}: Cidade e UF para o TR (usar o mesmo local do ETP).
{{data_tr}}: Data para o TR (usar a mesma data do ETP).
{{numero_processo_administrativo_tr}}: Número do processo administrativo (genérico, igual ao do ETP).
{{cnae_sugerido}}: Sugestão de CNAE.
{{prazo_vigencia_tr}}: Prazo de vigência (ex: 12 meses, alinhado com {prazos_estimados_py}).
{{regras_subcontratacao}}: Regras de subcontratação.
{{regras_garantia}}: Regras de garantia.
{{medicao_pagamento_dias_recebimento}}: Dias para recebimento.
{{medicao_pagamento_dias_faturamento}}: Dias para faturamento.
{{medicao_pagamento_dias_pagamento}}: Dias para pagamento.
{{criterio_julgamento_tr}}: Critério de julgamento.
{{metodologia_implementacao}}: Metodologia de implementação.
{{criterios_aceitacao_tr}}: Critérios de aceitação.
{{obrigações_contratado_tr}}: Obrigações do contratado.
{{obrigações_orgao_tr}}: Obrigações do orgão.
{{gestao_contrato_tr}}: Gestão do contrato.
{{sancoes_administrativas_tr}}: Sanções administrativas.
{{anexos_tr}}: Anexos.
{{esfera_administrativa}}: Esfera administrativa ({esfera_administrativa_py}).
MODELO DE ETP PARA PREENCHIMENTO (Siga esta estrutura rigorosamente):

Estudo Técnico Preliminar
Contratação de solução tecnológica para {titulo_projeto_py}

Processo Administrativo nº XXXXXX/{ano_actual_py}

{local_etp_full_py}

Histórico de Revisões
Data	Versão	Descrição	Autor
{today_formatted}	1.0	Finalização da primeira versão do documento	IA Xertica.ai
Área requisitante
Identificação da Área requisitante: {orgao_nome_py}
Nome do Responsável: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Matrícula: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]

Introdução
O Estudo Técnico Preliminar – ETP é o documento constitutivo da primeira etapa do planejamento de uma contratação, que caracteriza o interesse público envolvido e a sua melhor solução. Ele serve de base ao Termo de Referência a ser elaborado, caso se conclua pela viabilidade da contratação.

O ETP tem por objetivo identificar e analisar os cenários para o atendimento de demanda registrada no Documento de Formalização da Demanda – DFD, bem como demonstrar a viabilidade técnica e econômica das soluções identificadas, fornecendo as informações necessárias para subsidiar a tomada de decisão e o prosseguimento do respectivo processo de contratação.
{contexto_geral_orgao_py}

Referência: Inciso XI, do art. 2º e art. 11 da IN SGD/ME nº 94/2022.

Descrição do problema e das necessidade
A {orgao_nome_py} busca, por meio da iniciativa '{titulo_projeto_py}', endereçar um desafio crítico identificado: {justificativa_necessidade_py}. Este problema impede [descrever impacto negativo, ex: a eficiente prestação de serviços, a tomada de decisões ágeis, a otimização de recursos].

Necessidades do negócio
A contratação visa suprir as seguintes necessidades do negócio do {orgao_nome_py}, com impactos diretos na eficiência operacional e na entrega de serviços:

Redução de gargalos operacionais: Eliminar pontos de lentidão e ineficiência nos processos atuais, permitindo um fluxo de trabalho mais dinâmico.
Melhoria da experiência do cidadão e do usuário interno: Proporcionar canais de comunicação e acesso a serviços mais intuitivos, rápidos e satisfatórios, elevando os índices de aprovação.
Otimização da alocação de recursos: Liberar equipes e colaboradores de tarefas repetitivas e de baixo valor agregado, permitindo que se concentrem em atividades estratégicas e de maior impacto.
Tomada de decisões baseada em dados: Fornecer dashboards e relatórios analíticos que transformem grandes volumes de dados em inteligência acionável, subsidiando escolhas estratégicas e operacionais.
Garantia de conformidade e transparência: Assegurar que as operações estejam em total alinhamento com a legislação vigente e com os princípios de publicidade, promovendo a confiança e a integridade.
Requisitos da Contratação
Os requisitos gerais e específicos para a contratação da solução proposta são:

Requisitos Funcionais: A solução deve ser capaz de {objetivo_geral_py}, atuando de forma a [descrever como as funcionalidades chaves da solução Xertica.ai se conectam ao objetivo].
Requisitos Não Funcionais:
Segurança: A solução deve garantir a integridade, confidencialidade e disponibilidade dos dados, em conformidade com a LGPD e as melhores práticas de segurança da informação para o setor público.
Escalabilidade: Deve possuir capacidade de expandir seus recursos e funcionalidades para atender a um aumento futuro na demanda e no volume de dados, sem degradação de desempenho.
Disponibilidade: A solução deve operar com alta disponibilidade, minimizando interrupções e garantindo acesso contínuo aos serviços, com SLA (Service Level Agreement) de no mínimo 99.5%.
Integração: Deve ter flexibilidade para integrar-se com os sistemas legados e plataformas já utilizadas pelo {orgao_nome_py}, utilizando APIs ou outros protocolos de comunicação padrão.
Desempenho: A solução deve apresentar tempo de resposta ágil e eficiente, mesmo em cenários de alta demanda, assegurando uma experiência fluida para o usuário.
Aderência Tecnológica: A solução deve estar alinhada com as especificações detalhadas na proposta técnica da Xertica.ai, que aborda aspectos de arquitetura, stack tecnológica e compatibilidade.
Levantamento de mercado
O levantamento de mercado demonstrou que a contratação de soluções de Inteligência Artificial para otimização de processos e atendimento é uma tendência consolidada no setor público e privado. Diversas empresas oferecem ferramentas com capacidades variadas, mas a Xertica.ai se destaca por sua notória especialização em projetos para o setor público brasileiro. Possui um portfólio de aceleradores de IA focados em demandas governamentais e uma vivência comprovada em implementações complexas como [CITE EXEMPLOS COMO CASO PROCERGS, MPRS, TJES, se relevante ou invente um se não houver].

A análise de mercado evidenciou que a Xertica.ai oferece um diferencial competitivo significativo em termos de integração nativa com o Google Cloud Platform (se aplicável), agilidade na implantação via aceleradores e uma compreensão aprofundada das particularidades legais e operacionais da administração pública. Este diferencial garante não apenas a aderência técnica, mas também a segurança jurídica e a eficiência na entrega da solução, superando alternativas que poderiam apresentar custos ocultos com adaptação ou risco de não conformidade.

Estimativa de demanda - quantidade de bens e serviços
A estimativa de demanda para os serviços/bens objeto desta contratação será:

Quantitativos: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO, com base em volume esperado de usuários, transações ou dados]. Serão definidos em detalhe no Termo de Referência com base na análise do volume de {{estimativa_demanda}} e na capacidade de processamento dos aceleradores da Xertica.ai. Esta projeção levará em conta [aspectos como: número de servidores, volume de documentos a processar, interações esperadas com a ferramenta de IA] para garantir o dimensionamento adequado da infraestrutura e licenciamento.
Mapa comparativo dos custos
O mapa comparativo de custos detalhado está disponível na Proposta Comercial da Xertica.ai e foi ratificado por cuidadosa pesquisa de mercado e comparação com contratos similares. A análise não se restringiu apenas ao menor preço, mas considerou o Custo Total de Propriedade (TCO), o Retorno sobre Investimento (ROI) estimado, os custos de implementação, treinamento, suporte contínuo e a capacidade de inovação futura. O valor estimado foi ratificado por [CITE FONTES DE PESQUISA, PUBLICAÇÕES OU DISPENSA/INEXIGIBILIDADE SE PUDER OU DEIXE EM ABERTO]:

{price_map_to_use_template}

Estimativa de custo total da contratação
O valor estimado global para esta contratação, considerando todas as fases de implementação, licenciamento e suporte para o período de {prazos_estimados_py}, é de R$ {valor_estimado_input_final}. Este valor reflete a complexidade da solução, a especialização exigida e a garantia de resultados alinhados aos objetivos do {orgao_nome_py}.

Descrição da solução como um todo
A solução proposta abrange a implementação dos aceleradores de Inteligência Artificial da Xertica.ai, com foco em {nomes_produtos_selecionados_py}, que em conjunto formam uma plataforma integrada para [descrever o propósito geral da plataforma]. Esta solução é projetada para otimizar [mencionar processos específicos], usando [mencionar tecnologias chave da Xertica.ai, ex: processamento de linguagem natural, machine learning, visão computacional] para alcançar [mencionar benefícios específicos, ex: autonomia na análise de documentos, atendimento escalável, insights preditivos]. A proposta técnica da Xertica.ai (anexada) detalha a arquitetura, os componentes, os fluxos de trabalho e os serviços de implementação e suporte contínuo da solução, assegurando a aderência às necessidades do {orgao_nome_py}.

Justificativa do parcelamento ou não da contratação
Decisão sobre Parcelamento: {parcelamento_contratacao_py}.
Justificativa: {justificativa_parcelamento_final_py}.

Providências a serem tomadas
As providências a serem tomadas para a plena execução da contratação incluem:

Formalização do processo de contratação e assinatura do contrato.
Definição de cronograma detalhado de implantação e plano de projeto conjunto.
Designação de equipe técnica do {orgao_nome_py} para acompanhamento, validação e interface com a Xertica.ai.
Realização das etapas de capacitação e transferência de conhecimento intensiva para as equipes do {orgao_nome_py}.
Configuração e personalização da solução Xertica.ai, incluindo parametrização e integração com sistemas existentes do órgão.
Implantação em ambiente de produção, acompanhamento pós-implantação e fase de estabilização.
Acompanhamento contínuo da performance da solução e realização de ajustes finos e otimizações.
Declaração de viabilidade
Após a análise técnica e econômica preliminar, conjugada com a expertise da Xertica.ai no setor público e a aderência da solução aos objetivos estratégicos do {orgao_nome_py}, declara-se a viabilidade plena da contratação da solução Xertica.ai. A proposta demonstra ser a mais adequada, tecnológica, juridicamente sólida e economicamente vantajosa, alinhada às políticas e estratégias de modernização do {orgao_nome_py}, prometendo um retorno significativo sobre o investimento em termos de eficiência e qualidade dos serviços.

Responsáveis
A Equipe de Planejamento da Contratação foi instituída pela Portaria nº [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO] (ou outro instrumento equivalente de formalização), de {today_day_py} de {mes_extenso_py} de {ano_actual_py}.
Conforme o § 2º do Art. 11 da IN SGD/ME nº 94, de 2022, o Estudo Técnico Preliminar deverá ser aprovado e assinado pelos Integrantes Técnicos e Requisitantes e pela autoridade máxima da área de TIC.

INTEGRANTE TÉCNICO INTEGRANTE REQUISITANTE

[Nome do Integrante Técnico] [Nome do Integrante Requisitante]
Matrícula/SIAPE: [Matrícula/SIAPE] Matrícula/SIAPE: [Matrícula/SIAPE]

Aprovação e declaração de conformidade
Aprovo este Estudo Técnico Preliminar e atesto sua conformidade às disposições da Instrução Normativa SGD/ME nº 94, de 23 de dezembro de 2022.

{local_etp_full_py}

<NEWPAGE>
Termo de Referência – Nº [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
1 – DEFINIÇÃO DO OBJETO
A presente contratação tem como objeto a aquisição e implantação de uma solução de [descrever o conjunto dos aceleradores Xertica.ai selecionados] por meio da implementação da tecnologia da Xertica.ai, com o objetivo de {objetivo_geral_py}, garantindo a modernização e a otimização dos serviços do {orgao_nome_py}, conforme condições e exigências estabelecidas neste instrumento.

1.1. Objeto Sintético: Contratação de serviços estratégicos de Tecnologia da Informação baseados em Inteligência Artificial, caracterizados como [Bens e Serviços Comuns/Especiais – ESPECIFICAR AQUI COM BASE NA SOLUÇÃO, EX: SERVIÇOS TÉCNICOS ESPECIALIZADOS DE CONSULTORIA E DESENVOLVIMENTO DE IA], para atender às necessidades permanentes e contínuas de modernização da Administração Pública {esfera_administrativa_py}.

Ramo de Atividade predominante da contratação: {{cnae_sugerido}} - Consultoria em tecnologia da informação. O CNAE específico deverá ser confirmado e preenchido pelo órgão/administração.
Quantitativos estimados: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - Ex: 1 licença base, 500 usuários, 1000 transações/mês]. Serão definidos no contrato com base nas necessidades específicas e no dimensionamento da solução Xertica.ai que atenderá a demanda.
Prazo do contrato: O contrato terá vigência de [INFORMAR PRAZO, ex: 12 (doze) meses], contados a partir da assinatura, podendo ser prorrogado conforme a Lei nº 14.133/2021.
2 – FUNDAMENTAÇÃO DA CONTRATAÇÃO
A Fundamentação da Contratação e de seus quantitativos encontra-se pormenorizada em tópico específico dos Estudos Técnicos Preliminares (ETP), anexo a este Termo de Referência.

2.1. O objeto da contratação está previsto no Plano de Contratações Anual {ano_actual_py} do {orgao_nome_py}, conforme detalhamento a seguir:

ID PCA no PNCP e/ou SGA: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Data de publicação no PNCP: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Id do item no PCA: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
2.2. Justificativa da contratação: A contratação da solução Xertica.ai é fundamental para {justificativa_necessidade_py} no {orgao_nome_py}, buscando a otimização dos processos de trabalho e a melhoria contínua dos serviços prestados à sociedade. Esta solução de IA representa um avanço tecnológico que permitirá [descrever benefícios específicos, ex: automação de tarefas repetitivas, análise de dados em larga escala, atendimento customizado].

2.3. Enquadramento da contratação: A contratação fundamenta-se no {modelo_licitacao_py} e nas demais normas legais e regulamentares atinentes à matéria, em plena observância à Lei nº 14.133, de 1º de abril de 2021 (Nova Lei de Licitações e Contratos Administrativos). [SE FOR INEXIGIBILIDADE OU DISPENSA, CITAR ARTIGOS ESPECÍFICOS DA LEI 14.133/2021, ex: Art. 74, inciso II para notória especialização, ou Art. 75 para dispensa].

3 – DESCRIÇÃO DA SOLUÇÃO COMO UM TODO
A solução a ser contratada consiste na implementação e suporte dos aceleradores de Inteligência Artificial da Xertica.ai, com foco em {nomes_produtos_selecionados_py}, que se mostraram a alternativa mais vantajosa e completa para {objetivo_geral_py} no {orgao_nome_py}. Esta abordagem integrada garante a sinergia entre diferentes módulos e a entrega de um sistema coeso capaz de endereçar os desafios apresentados.

3.1. O objeto da contratação compreende:

A disponibilização, instalação e configuração dos aceleradores Xertica.ai selecionados, conforme detalhado na Proposta Técnica.
Serviços especializados de implementação, customização e integração da solução com os sistemas legados e plataformas existentes do {orgao_nome_py}.
Suporte técnico contínuo e especializado, bem como serviços de manutenção evolutiva e corretiva, garantindo a performance e a estabilidade da solução.
Treinamento e capacitação abrangente das equipes do {orgao_nome_py} para operação, administração e aproveitamento máximo das funcionalidades da ferramenta.
Garantia de atualização tecnológica e inovação, assegurando que a solução se mantenha moderna e aderente às futuras necessidades do órgão.
3.2. Forma de execução da contratação: indireta, em regime de [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - Ex: empreitada por preço global; preço unitário; tarefa; escopo. Escolha o mais apropriado para serviços de IA, geralmente EMPREITADA POR PREÇO GLOBAL OU ESCOPO].

3.3. A descrição detalhada da solução como um todo, incluindo arquitetura, funcionalidades e requisitos técnicos específicos, encontra-se pormenorizada no Estudo Técnico Preliminar e na Proposta Técnica da Xertica.ai (anexo).

4 – REQUISITOS DA CONTRATAÇÃO
Os requisitos necessários à contratação são essenciais para o atendimento da necessidade especificada e a garantia da qualidade, desempenho e segurança da solução.

4.1. Os requisitos necessários para a presente contratação são:

Requisitos Funcionais: A solução deve atender a todas as funcionalidades essenciais para {objetivo_geral_py}, conforme detalhado nas documentações dos aceleradores (Battle Cards e Data Sheets) e na aplicação específica descrita no ETP. Isso inclui [CITE AQUI 3-5 FUNCIONALIDADES CHAVES DA SOLUÇÃO, ex: capacidade de processamento de linguagem natural, análise preditiva de dados, automação de fluxos de trabalho, interface intuitiva].
Requisitos Não Funcionais:
Segurança: A solução deve estar em total conformidade com a Lei Geral de Proteção de Dados (LGPD) e demais regulamentações de segurança da informação aplicáveis ao setor público. Deve empregar criptografia de dados em trânsito e em repouso, mecanismos de autenticação robustos e controle de acesso baseado em perfis.
Escalabilidade: Deve possuir a capacidade de escalar horizontal e verticalmente para acomodar o crescimento futuro da demanda, do volume de usuários e de dados, sem comprometer o desempenho ou a estabilidade.
Disponibilidade: Garantia de alta disponibilidade da solução, com um Acordo de Nível de Serviço (SLA) estabelecido para no mínimo 99.5% de tempo de atividade (uptime).
Integração: A solução deve ser facilmente integrável com os sistemas e bases de dados existentes do {orgao_nome_py} (ex: APIs RESTful, Web Services, Banco de Dados SQL/NoSQL), garantindo um fluxo de dados contínuo e seguro.
Desempenho: Tempos de resposta ágeis para todas as operações, mesmo em picos de demanda, garantindo uma experiência de usuário satisfatória.
Manutenibilidade e Suporte: Facilidade de manutenção e atualização, com documentação técnica completa e suporte especializado disponível para resolução de incidentes e dúvidas.
Práticas de Sustentabilidade: Alinhamento com critérios de sustentabilidade ambiental, social e econômica na execução dos serviços e operação da solução, conforme diretrizes da Lei nº 14.133/2021.
4.2. Da exigência de carta de solidariedade: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - (SIM/NÃO) e justificativa, conforme edital. Se aplicável, a Xertica.ai pode fornecer.].

4.3. SUBCONTRATAÇÃO: {{regras_subcontratacao}} Apenas serviços pontuais e previamente autorizados pelo {orgao_nome_py} poderão ser subcontratados. (Ou: Não será permitida a subcontratação de partes relevantes do objeto da contratação; Ou: Totalmente Permitida; Ou: Condicionada à Aprovação - preencher conforme o caso).

4.4. GARANTIA DA CONTRATAÇÃO: {{regras_garantia}} A CONTRATADA deverá oferecer garantia técnica mínima de 12 (doze) meses sobre os serviços prestados e a solução implementada, contados a partir do recebimento definitivo do objeto. Esta garantia abrange correção de falhas e bom funcionamento da solução.

4.5. O Contratado deverá realizar a transição contratual com transferência de conhecimento, tecnologia e técnicas empregadas, assegurando a autonomia e capacidade do {orgao_nome_py} na operação e manutenção da solução a longo prazo.

4.6. VISTORIA: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - (SIM/NÃO) e detalhes].

5 – EXECUÇÃO DO OBJETO
A execução do objeto será realizada de forma indireta, com foco na entrega de resultados e na operação contínua e eficiente da solução.

5.1. O prazo de prestação dos serviços e entrega das soluções será definido nos termos da Ordem de Serviço, emitidas conforme cronograma detalhado na Proposta Técnica da Xertica.ai.

5.2. Os serviços serão executados e as soluções disponibilizadas predominantemente de forma remota, com possibilidade de visitas técnicas presenciais, se necessário e acordado entre as partes.

5.3. Deverão ser observados os métodos, rotinas e procedimentos de implementação e suporte definidos na Proposta Técnica da Xertica.ai, que seguirão as melhores práticas de gestão de projetos de TI. {{metodologia_implementacao}}

5.4. A CONTRATADA deverá disponibilizar todos os materiais, equipamentos, softwares e ferramentas necessárias para a perfeita execução dos serviços e funcionamento da solução.

5.5. O prazo de garantia contratual dos serviços será de, no mínimo, 12 (doze) meses, contado a partir do recebimento definitivo do objeto, conforme Art. 99 da Lei nº 14.133/2021.

6 – GESTÃO DO CONTRATO
{{gestao_contrato_tr}}
A gestão do contrato será realizada por um fiscal do {orgao_nome_py}, que será responsável por acompanhar a execução do objeto, verificar o cumprimento das obrigações e aprovar os pagamentos, conforme as normas aplicáveis.

6.1. O contrato deverá ser executado fielmente pelas partes, de acordo com as cláusulas avençadas e as rigorosas normas da Lei nº 14.133, de 2021.
6.2. As comunicações oficiais entre o órgão e a contratada devem ser realizadas por escrito, preferencialmente via sistema de gerenciamento de contratos ou e-mail institucional, garantindo a rastreabilidade.
6.3. O CONTRATANTE poderá convocar representante da empresa para adoção de providências imediatas em caso de não conformidade ou necessidade de ajustes urgentes.
6.4. A formalização da contratação ocorrerá por meio de termo de contrato, o qual será publicado no PNCP e no portal de transparência do {orgao_nome_py}.
6.5. Após a assinatura, haverá reunião inicial para alinhamento e apresentação do plano de fiscalização detalhado por parte do {orgao_nome_py}.
6.6. A execução do contrato será acompanhada e fiscalizada continuamente por fiscal(is) do contrato, designados formalmente, com poderes para atestar o cumprimento das obrigações.
6.7. A CONTRATADA deverá manter preposto para representá-la na execução do contrato, sendo o ponto focal para todas as questões técnico-operacionais e administrativas.

7 – MEDIÇÃO E PAGAMENTO
A medição e pagamento serão realizados com base na entrega e no atingimento dos Acordos de Níveis de Serviço (ANS) ou Instrumentos de Medição de Resultados (IMR) associados à solução.

7.1. A avaliação da execução do objeto utilizará o Instrumento de Medição de Resultado (IMR), conforme prescrições: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - metodologia e regras, ex: cumprimento de SLAs, disponibilidade da solução, satisfação do usuário, performance de processamento]. {{criterios_aceitacao_tr}}
7.2. O valor devido a título de pagamento mensal à CONTRATADA será mensurado pelos indicadores do IMR, garantindo o alinhamento do pagamento à performance e aos resultados alcançados.
7.3. O recebimento dos serviços será provisório e definitivo, conforme Art. 140 da Lei nº 14.133/2021, atestando a qualidade e a conformidade da entrega.
7.4. Do Faturamento: A CONTRATADA deverá apresentar fatura ou nota fiscal no prazo de 5 (cinco) dias úteis após comunicação formal do gestor do contrato sobre o ateste da medição, juntamente com as comprovações de regularidade fiscal e trabalhista.
7.5. Das condições de pagamento: O pagamento será efetuado em 15 (quinze) dias corridos após o ateste da Fatura/Nota Fiscal, condicionado à verificação de conformidade na prestação dos serviços.

8 – SELEÇÃO DO FORNECEDOR
A seleção do fornecedor será realizada conforme a modalidade de contratação definida, garantindo a lisura, a transparência e a adequação legal do processo.

8.1. O fornecedor será selecionado por meio de {modelo_licitacao_py}, com adoção do critério de julgamento pelo [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - Ex: MENOR PREÇO, MELHOR TÉCNICA, MELHOR TÉCNICA E PREÇO, MAIOR RETORNO ECONÔMICO]. {{criterio_julgamento_tr}}

Exigências de Habilitação: Serão observados os requisitos exigidos no Aviso de Contratação ou Edital, em conformidade com o Art. 62 a 70 da Lei nº 14.133/2021, abrangendo habilitação jurídica, qualificação técnica, qualificação econômico-financeira, regularidade fiscal e trabalhista e cumprimento do disposto no inciso XXXIII do art. 7º da Constituição Federal.
9 – ESTIMATIVA DO PREÇO
A estimativa de preço baseia-se na Proposta Comercial da Xertica.ai e em detalhada análise de mercado, sendo um valor de referência para a contratação que reflete a qualidade e a inovação tecnológica da solução.

9.1. O valor estimado da contratação é de R$ {valor_estimado_input_final}, conforme detalhado no Estudo Técnico Preliminar. A memória de cálculo e a composição dos custos estão disponíveis em documento separado, embasadas em levantamentos de preços de mercado e propostas anteriores.

10 – ADEQUAÇÃO ORÇAMENTÁRIA
A adequação orçamentária para a presente contratação será assegurada pelos recursos consignados na Lei Orçamentária Anual do {esfera_administrativa_py} ({orgao_nome_py}), garantindo a cobertura das despesas pelo período de vigência.

10.1. As despesas decorrentes da presente contratação correrão à conta de recursos específicos consignados no Orçamento Geral da {esfera_administrativa_py}, mediante a seguinte dotação:

UG Executora: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Programa de Trabalho: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Fonte: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]
Natureza da Despesa: [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - ex: 44.90.39 - Outros Serviços de Terceiros - Pessoa Jurídica]
10.2. A dotação relativa aos exercícios financeiros subsequentes será indicada após aprovação da Lei Orçamentária respectiva e liberação dos créditos correspondentes, mediante apostilamento, conforme previsão legal.

Há anexos no pedido: Sim (Proposta Comercial e Proposta Técnica da Xertica.ai, Battle Cards e Data Sheets dos aceleradores, e outros documentos de contexto legal).
""".format(
esfera_administrativa_py=esfera_administrativa_py,
orgao_nome_py=orgao_nome_py,
titulo_projeto_py=titulo_projeto_py,
justificativa_necessidade_py=justificativa_necessidade_py,
objetivo_geral_py=objetivo_geral_py,
prazos_estimados_py=prazos_estimados_py,
valor_estimado_input_final=valor_estimado_input_final,
modelo_licitacao_py=modelo_licitacao_py,
parcelamento_contratacao_py=parcelamento_contratacao_py,
justificativa_parcelamento_final_py=justificativa_parcelamento_final_py,
contexto_geral_orgao_py=(contexto_geral_orgao_py if contexto_geral_orgao_py else f"A {orgao_nome_py}, em sua missão de modernizar a gestão pública e aprimorar a prestação de serviços ao cidadão, busca constantemente soluções inovadoras que garantam eficiência, transparência e segurança."),
llm_context_data_json=llm_context_data_json,
proposta_comercial_content_py=proposta_comercial_content_py,
proposta_tecnica_content_py=proposta_tecnic_content_py,
price_map_to_use_template=price_map_to_use_template,
gcs_accel_str=gcs_accel_str,
gcs_legal_str=gcs_legal_str,
today_formatted=today_formatted,
mes_extenso_py=mes_extenso_py,
ano_actual_py=ano_actual_py,
local_etp_full_py=local_etp_full_py,
nomes_produtos_selecionados_py=nomes_produtos_selecionados_py,
)

try:
    logger.info(f"Enviando prompt para o Gemini (primeiros 500 chars): {llm_prompt_content_final[:500]}...")
    response = await gemini_model.generate_content_async(
        llm_prompt_content_final,
        generation_config=_generation_config
    )
    
    if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
        logger.error("Resposta do Gemini inválida ou sem conteúdo esperado.")
        logger.error(f"Resposta completa do Gemini: {response}")
        raise Exception("Resposta inválida do modelo Gemini (sem partes de conteúdo).")

    generated_text = response.candidates[0].content.parts[0].text
    logger.info(f"Resposta RAW do Gemini (primeiros 2000 chars): {generated_text[:2000]}...")
    
    if generated_text:
        match_json = re.search(r"```json\s*([\s\S]*?)\s*```", generated_text, re.DOTALL)
        if match_json:
            json_str = match_json.group(1)
            logger.info("JSON extraído de bloco de código Markdown.")
        else:
            json_str = generated_text
        
        parsed_content = json.loads(json_str)
        logger.info("Resposta do Gemini recebida e parseada como JSON com sucesso.")
        return parsed_content
    else:
        logger.warning("Resposta do Gemini vazia ou não contém texto processável.")
        raise Exception("Resposta vazia ou não processável do modelo Gemini.")

except json.JSONDecodeError as e:
    logger.error(f"Erro ao parsear JSON da resposta do Gemini: {e}.")
    problematic_json_string = generated_text if 'generated_text' in locals() else "Não foi possível capturar a string problemática."
    logger.error(f"String JSON que causou o erro de parse: {problematic_json_string}")
    raise HTTPException(status_code=500, detail=f"Erro no formato JSON retornado pelo Gemini: {e}. Verifique os logs do servidor para a string exata.")
except AttributeError as e:
    logger.error(f"Estrutura da resposta do Gemini inesperada: {e}")
    logger.error(f"Resposta completa do Gemini: {response}")
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
prazosEstimados: str = Form(..., description="Prazos estimados para implantação e execução. Ex: 3 meses para implantação, 12 meses de operação."),
modeloLicitacao: str = Form(..., description="Modelo de licitação pretendido (ex: Pregão Eletrônico, Inexigibilidade, Dispensa)."),
parcelamentoContratacao: str = Form(..., description="Indica se a contratação será parcelada (Sim, Não, Justificar)."),
contextoGeralOrgao: Optional[str] = Form(None, description="Breve contexto sobre o órgão, seus desafios e iniciativas relevantes."),
valorEstimado: Optional[float] = Form(None, description="Valor total estimado da contratação (opcional)."),
justificativaParcelamento: Optional[str] = Form(None, description="Justificativa caso o parcelamento seja 'Justificar' ou 'Não'."),
propostaComercialFile: Optional[UploadFile] = File(None, description="Proposta Comercial da Xertica.ai em PDF (opcional)."),
propostaTecnicaFile: Optional[UploadFile] = File(None, description="Proposta Técnica da Xertica.ai em PDF (opcional).")
):
"""
Endpoint principal para gerar os documentos ETP e TR.
Recebe dados do formulário, processa arquivos PDF (se enviados), consulta dados contextuais do GCS,
chama o LLM Gemini para gerar o conteúdo dos documentos e, por fim,
cria um novo Google Doc com o conteúdo gerado e retorna o link.
"""
logger.info(f"Requisição para gerar ETP/TR para '{tituloProjeto}' do órgão '{orgaoSolicitante}' recebida.")

if not gemini_model:
    raise HTTPException(status_code=503, detail="Serviço de IA (LLM) indisponível ou não configurado.")
if not storage_client:
    raise HTTPException(status_code=503, detail="Serviço de Armazenamento (GCS) indisponível ou não configurado.")

form_data = await request.form()
# produtosXertica_list agora contém os nomes originais dos produtos (ex: "Meet Transcriber")
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
    "valorEstimado": valorEstimado,
    "justificativaParcelamento": justificativaParcelamento if justificativaParcelamento else "Não fornecida.",
    "produtosXertica": produtosXertica_list, # Contém nomes originais (ex: "Meet Transcriber")
    "data_geracao_documento": date.today().strftime("%d/%m/%Y"),
}

for product_original_name in produtosXertica_list: # product_original_name é o valor da option no HTML
    # A chave de integração vem do JS, usando o nome normalizado no `name` do textarea
    # Ex: name="integracao_Meet_Transcriber"
    product_normalized_for_form_key = product_original_name.replace(' ', '_').replace('.', '_')
    integration_key_expected_by_llm_prompt = f"integracao_{product_normalized_for_form_key}"
    
    llm_context_data[integration_key_expected_by_llm_prompt] = form_data.get(integration_key_expected_by_llm_prompt, f"Detalhes de integração para {product_original_name} não fornecidos.")
    logger.info(f"Detalhe de integração para '{product_original_name}': {llm_context_data[integration_key_expected_by_llm_prompt]}")


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

llm_context_data['gcs_accelerator_content'] = {}
llm_context_data['gcs_legal_context_content'] = {}

for product_original_name in produtosXertica_list:
    # product_original_name é o valor da tag <option value="..."> (ex: "Meet Transcriber")
    # Criamos o nome normalizado para o caminho do GCS
    product_normalized_for_gcs_path = product_original_name.replace(' ', '_').replace('.', '_')

    # Casos especiais para GCP, GMP, GWS (assumindo que o nome do produto selecionado é "GCP", "GMP", "GWS")
    if product_original_name == "GCP":
        content_op_gcp = get_gcs_file_content(f"GCP/Análise_Técnica_Google_Cloud_Platform_.txt")
        if content_op_gcp: llm_context_data['gcs_accelerator_content'][f"{product_original_name}_OP_GCP"] = content_op_gcp
        else: logger.warning(f"GCP analysis text not found at GCP/Análise_Técnica_Google_Cloud_Platform_.txt")
        continue

    if product_original_name == "GMP":
        content_op_gmp = get_gcs_file_content(f"GMP/Google_Maps_Platform_Análise_Técnica_.txt")
        if content_op_gmp: llm_context_data['gcs_accelerator_content'][f"{product_original_name}_OP_GMP"] = content_op_gmp
        else: logger.warning(f"GMP analysis text not found at GMP/Google_Maps_Platform_Análise_Técnica_.txt")
        continue

    if product_original_name == "GWS":
        content_op_gws = get_gcs_file_content(f"GWS/Análise_técnica_do_Google_Workspace_.txt")
        if content_op_gws: llm_context_data['gcs_accelerator_content'][f"{product_original_name}_OP_GWS"] = content_op_gws
        else: logger.warning(f"GWS analysis text not found at GWS/Análise_técnica_do_Google_Workspace_.txt")
        continue
    
    # Para os aceleradores padrão que seguem o formato BC/DS/OP
    # Tentativa 1: Pasta/Arquivo com nome normalizado (ex: aceleradores_conteudo/Meet_Transcriber/BC_Meet_Transcriber.txt)
    bc_path_norm_norm = f"aceleradores_conteudo/{product_normalized_for_gcs_path}/BC_{product_normalized_for_gcs_path}.txt"
    ds_path_norm_norm = f"aceleradores_conteudo/{product_normalized_for_gcs_path}/DS_{product_normalized_for_gcs_path}.txt"
    op_path_norm_norm = f"aceleradores_conteudo/{product_normalized_for_gcs_path}/OP_{product_normalized_for_gcs_path}.txt"
    
    # Tentativa 2: Pasta normalizada, Arquivo com nome original (ex: aceleradores_conteudo/Meet_Transcriber/BC - Meet Transcriber.txt)
    bc_path_norm_orig = f"aceleradores_conteudo/{product_normalized_for_gcs_path}/BC - {product_original_name}.txt"
    ds_path_norm_orig = f"aceleradores_conteudo/{product_normalized_for_gcs_path}/DS - {product_original_name}.txt"
    op_path_norm_orig = f"aceleradores_conteudo/{product_normalized_for_gcs_path}/OP - {product_original_name}.txt"
    
    content_bc = get_gcs_file_content(bc_path_norm_norm) or get_gcs_file_content(bc_path_norm_orig)
    content_ds = get_gcs_file_content(ds_path_norm_norm) or get_gcs_file_content(ds_path_norm_orig)
    content_op = get_gcs_file_content(op_path_norm_norm) or get_gcs_file_content(op_path_norm_orig)

    # As chaves para o LLM usam o nome original do produto (ex: "Meet Transcriber_BC")
    if content_bc: llm_context_data['gcs_accelerator_content'][f"{product_original_name}_BC"] = content_bc
    if content_ds: llm_context_data['gcs_accelerator_content'][f"{product_original_name}_DS"] = content_ds
    if content_op: llm_context_data['gcs_accelerator_content'][f"{product_original_name}_OP"] = content_op
    logger.info(f"Conteúdo GCS para acelerador padrão '{product_original_name}': BC {'encontrado' if content_bc else 'não encontrado'}, DS {'encontrado' if content_ds else 'não encontrado'}, OP {'encontrado' if content_op else 'não encontrado'}.")


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

logger.info("Enviando dados de contexto para o LLM Gemini.")
llm_response = await generate_etp_tr_content_with_gemini(llm_context_data)

document_subject = llm_response.get("subject", f"ETP e TR: {orgaoSolicitante} - {tituloProjeto} ({date.today().strftime('%Y-%m-%d')})")
etp_content_md = llm_response.get("etp_content", "# ETP\\n\\nErro: Conteúdo do ETP não foi gerado corretamente pelo LLM.")
tr_content_md = llm_response.get("tr_content", "# Termo de Referência\\n\\nErro: Conteúdo do TR não foi gerado corretamente pelo LLM.")

docs_service, drive_service = authenticate_google_docs_and_drive()

try:
    new_doc_body = {'title': document_subject}
    new_doc_metadata = drive_service.files().create(body=new_doc_body, fields='id,webViewLink').execute()
    document_id = new_doc_metadata.get('id')
    document_link_initial = new_doc_metadata.get('webViewLink')
    
    if not document_id:
        logger.error("Falha ao criar novo documento no Google Docs. ID não retornado.")
        raise HTTPException(status_code=500, detail="Falha ao criar novo documento no Google Docs (ID não obtido).")
    logger.info(f"Documento Google Docs criado com ID: {document_id}, Link inicial: {document_link_initial}")

    combined_markdown_content = f"{etp_content_md}\n\n<NEWPAGE>\n\n{tr_content_md}" 
    
    requests_for_docs_api = apply_basic_markdown_to_docs_requests(combined_markdown_content)
    
    if requests_for_docs_api:
        docs_service.documents().batchUpdate(
            documentId=document_id,
            body={'requests': requests_for_docs_api}
        ).execute()
        logger.info(f"Conteúdo ETP e TR inserido e formatado no documento Google Docs: {document_id}")
    else:
        logger.warning(f"Nenhuma request de formatação gerada para o documento {document_id}. Conteúdo pode estar vazio ou não formatado.")

    permission = {'type': 'anyone', 'role': 'reader'}
    try:
        drive_service.permissions().create(fileId=document_id, body=permission, fields='id').execute()
        logger.info(f"Permissões de leitura pública definidas para o documento: {document_id}")
    except HttpError as e_perm:
        logger.warning(f"Não foi possível aplicar permissão 'anyone' ao documento {document_id}: {e_perm}. "
                       f"O documento pode não ser publicamente acessível. Verifique as políticas do domínio.")

    file_metadata_final = drive_service.files().get(fileId=document_id, fields='webViewLink, id').execute()
    document_link_final = file_metadata_final.get('webViewLink')

    if not document_link_final:
        logger.error(f"Falha ao obter o link final do Google Docs para o documento {document_id}.")
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
    error_content = e_google_api.content.decode() if e_google_api.content else "{}"
    try:
        error_details_json = json.loads(error_content)
        error_message = error_details_json.get('error', {}).get('message', f"Erro desconhecido na API do Google. Status: {e_google_api.resp.status}")
    except json.JSONDecodeError:
        error_message = f"Erro na API do Google (resposta não JSON): {error_content}. Status: {e_google_api.resp.status}"
    
    logger.exception(f"Erro na API do Google Docs/Drive durante criação/atualização do documento: {error_message}") 
    raise HTTPException(status_code=e_google_api.resp.status if hasattr(e_google_api, 'resp') else 500, 
                        detail=f"Erro na API do Google Docs/Drive: {error_message}")
except Exception as e:
    logger.exception(f"Erro inesperado durante a geração ou criação do documento Google Docs: {e}") 
    raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no servidor: {e}. Verifique os logs.")
