import os
import logging
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Google Cloud Imports
from google.cloud import storage
# from google.cloud import aiplatform # Descomentar para usar Vertex AI (Gemini)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json

# Para simulação de LLM, e futura integração real
# from vertexai.preview.generative_models import GenerativeModel # Exemplo de import para Gemini
# import vertexai # Exemplo para inicialização do Vertex AI

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Carrega variáveis de ambiente (para desenvolvimento local) ---
load_dotenv()

app = FastAPI(
    title="Gerador de ETP e TR Xertica.ai",
    description="Backend inteligente para gerar documentos ETP e TR.",
    version="0.1.0"
)

# --- Configurações CORS ---
# Permite que seu frontend (se estiver em outro domínio) acesse o backend.
# Em produção, substitua "*" pelos domínios específicos do seu frontend.
origins = ["*"] # Ideal para desenvolvimento. Em PROD, restrinja!
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variáveis de Ambiente e Inicialização de Clientes GCP ---
# Certifique-se que estas variáveis estejam setadas no ambiente do Cloud Run!
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "docsorgaospublicos")

if not GCP_PROJECT_ID:
    logger.error("GCP_PROJECT_ID não está configurado. Verifique as variáveis de ambiente do Cloud Run.")
    # Em um cenário de produção mais rígido, levantaria uma exceção aqui.

try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Google Cloud Storage client inicializado.")
except Exception as e:
    logger.exception("Erro ao inicializar Google Cloud Storage client.")
    # A aplicação pode continuar, mas a leitura do GCS falhará.

# --- Funções Auxiliares para GCP ---

def authenticate_google_docs_and_drive():
    """
    Autentica com as APIs do Google Docs e Drive.
    No Cloud Run, utiliza as credenciais da Service Account do próprio serviço (ADC - Application Default Credentials).
    """
    try:
        docs_service = build('docs', 'v1', cache_discovery=False)
        drive_service = build('drive', 'v3', cache_discovery=False)
        logger.info("Google Docs e Drive services inicializados.")
        return docs_service, drive_service
    except Exception as e:
        logger.exception("Erro ao autenticar/inicializar Google Docs/Drive APIs.")
        raise HTTPException(status_code=500, detail=f"Falha na autenticação da API do Google Docs/Drive: {e}")

def get_gcs_file_content(file_path: str) -> Optional[str]:
    """Lê o conteúdo de um arquivo de texto de um bucket GCS."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        if blob.exists():
            content = blob.download_as_text()
            logger.info(f"Conteúdo de GCS://{GCS_BUCKET_NAME}/{file_path} lido com sucesso ({len(content)} chars).")
            return content
        logger.warning(f"Arquivo não encontrado no GCS: {file_path}")
        return None
    except Exception as e:
        logger.exception(f"Erro ao ler arquivo GCS {file_path}.")
        return None

async def upload_file_to_gcs(upload_file: UploadFile, destination_path: str) -> str:
    """Faz upload de um arquivo para o GCS e retorna o URL público."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_path)
        
        contents = await upload_file.read() # Ler o conteúdo do UploadFile

        blob.upload_from_string(contents, content_type=upload_file.content_type)
        blob.make_public() # Torna o objeto publicamente acessível via URL
        logger.info(f"Arquivo '{upload_file.filename}' carregado para GCS://{GCS_BUCKET_NAME}/{destination_path} e tornado público.")
        return blob.public_url
    except Exception as e:
        logger.exception(f"Erro ao fazer upload do arquivo '{upload_file.filename}' para GCS.")
        raise HTTPException(status_code=500, detail=f"Falha ao carregar arquivo: {e}")

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """
    Função mock para extrair texto de PDF.
    EM UMA IMPLEMENTAÇÃO REAL:
    - Usaria Google Cloud Document AI (recomedado para precisão e complexidade)
    - OU Google Cloud Vision AI (para OCR em PDFs escaneados)
    - OU Bibliotecas Python como `pypdf`, `pdfminer.six` (para PDFs text-searchable).
    """
    logger.info(f"Simulando extração de texto de PDF: {pdf_file.filename}")
    # Conteúdo real do PDF seria lido aqui:
    # content = await pdf_file.read()
    # text = seu_extrator_de_pdf(content)
    # Exemplo:
    # from pypdf import PdfReader
    # reader = PdfReader(io.BytesIO(content))
    # text = ""
    # for page in reader.pages:
    #     text += page.extract_text()
    
    return f"Conteúdo extraído da proposta (SIMULADO do PDF: {pdf_file.filename})" \
           f" - Este texto seria analisado pelo Vertex AI/Gemini para gerar o documento." \
           f" Ex.: Esta proposta comercial detalha [serviços], [prazos] e tem valor estimado de [Valor]."

# --- LÓGICA CENTRAL: SIMULAÇÃO DO LLM (COM PROMPT IMPECÁVEL) ---
def simulate_llm_response(prompt: str, context_data: Dict) -> Dict:
    """
    Simula a resposta de um LLM (como Google Gemini via Vertex AI) para gerar o conteúdo do ETP/TR.
    Este é o local onde a chamada à API REAL do LLM seria feita.
    """
    logger.info(f"Iniciando simulação LLM. Prompt length: {len(prompt)} chars. Context keys: {list(context_data.keys())}")

    # --- SIMULAÇÃO DA INFERÊNCIA E GERAÇÃO DE CONTEÚDO PELO LLM ---
    # Imagine que o LLM recebeu o 'prompt' completo com todos os dados do formulário e contextos do GCS/PDFs.
    # Ele agora processa essa informação para gerar o ETP e TR de forma inteligente.

    # Inferência para as seções ETP (Necessidade, Solução, Justificativa Contratual, Análise de Riscos, Benefícios, Cronograma, Valor Estimado)
    problem = context_data.get("justificativaNecessidade", "um problema complexo no órgão.")
    objective = context_data.get("objetivoGeral", "alcançar novos patamares de eficiência.")
    orgao_nome = context_data.get('orgaoSolicitante', 'O Órgão Solicitante')
    titulo_projeto = context_data.get('tituloProjeto', 'uma nova iniciativa')

    accelerator_benefits_summary = []
    for product_name in context_data.get("produtosXertica", []):
        # A IA "leria" os BC/DS/OP para extrair estes detalhes
        product_desc_from_bc = context_data.get(f"{product_name}_BC.txt", "") # Conteúdo real do BC do GCS
        # Aqui, simula a sumarização ou extração de pontos chave pelo LLM
        simulated_product_summary = f"O acelerador **{product_name}** da Xertica.ai oferece funcionalidades inovadoras para {problem.lower()}, transformando os desafios em oportunidades. Baseado em avançadas técnicas de IA, visa {objective.lower()} com agilidade e precisão."
        
        # Se houver integração específica do usuário, o LLM a usaria para refinar a descrição
        customer_specific_integration = context_data.get(f"integracao_{product_name}", "").strip()
        if customer_specific_integration:
            simulated_product_summary += f" Com foco especial na aplicação: '{customer_specific_integration}', garantindo que a solução se ajuste perfeitamente às necessidades operacionais de {orgao_nome}."

        accelerator_benefits_summary.append(simulated_product_summary)

    # Conteúdo das propostas, que o LLM "leu" (simuladamente)
    proposta_comercial_text = context_data.get("proposta_comercial_content", "Não foi possível extrair conteúdo da proposta comercial.")
    proposta_tecnica_text = context_data.get("proposta_tecnica_content", "Não foi possível extrair conteúdo da proposta técnica.")

    # Justificativa Legal - LLM inferiria com base no modeloLicitacao e contexto legal (mti/mpap)
    contracting_model = context_data.get("modeloLicitacao", "Não Informado")
    legal_justification_llm_simulated = ""
    if "Art. 75 Contratacao via MTI" == contracting_model:
        legal_justification_llm_simulated = f"A contratação via MTI justifica-se no Art. 75, IV, 'e' da Lei nº 14.133/2021, por envolver contratação com entidade da Administração Pública para atender a uma necessidade institucional, otimizando recursos e aproveitando o know-how existente."
    elif "Adesao a ATA do MPAP" == contracting_model:
        legal_justification_llm_simulated = f"A adesão à Ata de Registro de Preços do MPAP está fundamentada no Art. 86 da Lei nº 14.133/2021 e no Decreto nº 11.462/2023, proporcionando celeridade e economicidade pela utilização de condições já consolidadas."
    elif "Inexigibilidade via Declaracao ABES" == contracting_model:
        legal_justification_llm_simulated = f"A inexigibilidade de licitação, conforme o Art. 74 da Lei nº 14.133/2021, aplica-se à inviabilidade de competição. A declaração da ABES é crucial para atestar a exclusividade da solução de IA proposta pela Xertica.ai, confirmando a essencialidade e singularidade do objeto."
    else:
        legal_justification_llm_simulated = f"O modelo de '{contracting_model}' será aplicado conforme os artigos pertinentes da Lei nº 14.133/2021, garantindo a seleção da proposta mais vantajosa para a Administração Pública."

    # Análise de Riscos - LLM geraria com base no conhecimento ou no arquivo de risco do GCS.
    risk_analysis_llm_simulated = """
    **Análise Preliminar de Riscos e Mitigação:**
    1.  **Risco de Adaptação Tecnológica:** Implementação de novas tecnologias pode requerer adaptação.
        *   **Mitigação:** Suporte técnico dedicado da Xertica.ai e programas de treinamento abrangentes.
    2.  **Risco de Segurança de Dados:** Manuseio de informações sensíveis requer robustez.
        *   **Mitigação:** Soluções Xertica.ai desenvolvidas com princípios de Privacy by Design e Security by Default, em conformidade com a LGPD e melhores práticas de mercado.
    3.  **Risco de Interrupção de Serviço:** Eventuais falhas na infraestrutura de nuvem.
        *   **Mitigação:** Arquitetura de alta disponibilidade no GCP, redundância e planos de contingência, com monitoramento ativo e SLAs claros.
    """
    
    # Construção do ETP (em Markdown, para fácil inserção no Docs)
    etp_content = f"""
# ESTUDO TÉCNICO PRELIMINAR (ETP)

## 1. Necessidade da Contratação e Problema a ser Resolvido

O {orgao_nome} busca, por meio da iniciativa '{titulo_projeto}', endereçar um desafio crítico identificado: **{problem}**.
{context_data.get('contextoGeralOrgao', '')}

## 2. Objetivo da Contratação

O principal objetivo a ser alcançado com esta contratação é: **{objective}**. Espera-se que esta solução traga avanços significativos na gestão e operação do {orgao_nome}.

## 3. Solução Proposta - Aceleradores Xertica.ai

Para atender à necessidade e ao objetivo, propõe-se a implementação das seguintes soluções de Inteligência Artificial da Xertica.ai, que se alinham perfeitamente ao contexto do órgão:

{chr(10).join(accelerator_benefits_summary)}

## 4. Modelo de Contratação e Justificativa Legal

A contratação será realizada sob a modalidade de **{contracting_model}**.
**Justificativa Legal:** {legal_justification_llm_simulated}

## 5. Análise Preliminar de Riscos e Mitigação

{risk_analysis_llm_simulated}

## 6. Cronograma Estimado

Os prazos estimados para implantação e execução da solução são: **{context_data.get('prazosEstimados')}**.

## 7. Valor Estimado da Contratação

O valor estimado global para esta contratação é de **R$ {context_data.get('valorEstimado', 'a ser definido em detalhe no orçamento')}**.

## 8. Aspectos do Parcelamento (se aplicável)

**Decisão sobre Parcelamento:** **{context_data.get('parcelamentoContratacao')}**.
**Justificativa:** {context_data.get('justificativaParcelamento', 'Não se aplica.')}

"""

    # Construção do TR (em Markdown)
    tr_content = f"""
# TERMO DE REFERÊNCIA (TR)

## 1. Objeto da Contratação

O presente Termo de Referência (TR) tem por objeto a contratação de serviços e soluções de Inteligência Artificial da Xertica.ai, com o foco em **{problem.lower()}** e com o objetivo de **{objective.lower()}**, por meio da implementação dos aceleradores Xertica.ai.

## 2. Especificações Técnicas Detalhadas das Soluções

A solução compreenderá a disponibilização, customização e suporte dos seguintes aceleradores Xertica.ai, detalhados em suas características técnicas e benefícios esperados:

{chr(10).join(accelerator_benefits_summary)}

## 3. Regime de Execução

A execução do objeto do contrato se dará conforme o regime de **{contracting_model}**.

## 4. Condições e Prazos de Execução

Os prazos para o fornecimento, implantação e operação das soluções Xertica.ai são de **{context_data.get('prazosEstimados')}**. Detalhes específicos de marcos e entregas serão definidos no plano de trabalho anexo ao contrato.

## 5. Propostas e Anexos

São partes integrantes e complementares deste Termo de Referência as propostas submetidas pela Xertica.ai:
- **Proposta Comercial:** Conforme documento PDF anexado ao processo.
- **Proposta Técnica:** Conforme documento PDF anexado ao processo.

"""
    
    # O LLM retornaria um JSON como este
    return {
        "subject": f"ETP e TR - {titulo_projeto} - {orgao_nome}",
        "etp_content": etp_content,
        "tr_content": tr_content
    }

# --- Endpoint FastAPI ---

@app.post("/generate_etp_tr")
async def generate_etp_tr_endpoint(
    request: Request, # Usamos Request para acessar form() e obter lista de produtos
    orgaoSolicitante: str = Form(...),
    tituloProjeto: str = Form(...),
    justificativaNecessidade: str = Form(...),
    objetivoGeral: str = Form(...),
    prazosEstimados: str = Form(...),
    modeloLicitacao: str = Form(...),
    parcelamentoContratacao: str = Form(...),
    contextoGeralOrgao: Optional[str] = Form(None),
    valorEstimado: Optional[float] = Form(None),
    justificativaParcelamento: Optional[str] = Form(None),
    propostaComercialFile: Optional[UploadFile] = File(None),
    propostaTecnicaFile: Optional[UploadFile] = File(None)
):
    logger.info(f"Requisição recebida para {tituloProjeto} de {orgaoSolicitante}.")

    # Coletar todos os dados do formulário
    # getlist para campos de múltiplos valores (como select multiple)
    form_data = await request.form()
    produtosXertica_list = form_data.getlist("produtosXertica")

    llm_context_data = {
        "orgaoSolicitante": orgaoSolicitante,
        "tituloProjeto": tituloProjeto,
        "justificativaNecessidade": justificativaNecessidade,
        "objetivoGeral": objetivoGeral,
        "prazosEstimados": prazosEstimados,
        "modeloLicitacao": modeloLicitacao,
        "parcelamentoContratacao": parcelamentoContratacao,
        "contextoGeralOrgao": contextoGeralOrgao,
        "valorEstimado": valorEstimado,
        "justificativaParcelamento": justificativaParcelamento,
        "produtosXertica": produtosXertica_list,
    }

    # Adicionar os campos dinâmicos "integracao_[produto]"
    for key in form_data.keys():
        if key.startswith("integracao_"):
            llm_context_data[key] = form_data.get(key)
    
    # --- Processar e Unificar Conteúdo de Propostas PDF ---
    if propostaComercialFile and propostaComercialFile.filename:
        # Em um cenário real, você faria await extract_text_from_pdf(propostaComercialFile)
        # e passaria o texto extraído para o LLM.
        # Por enquanto, mantemos o upload para GCS e um conteúdo simulado.
        llm_context_data["proposta_comercial_content"] = await extract_text_from_pdf(propostaComercialFile)
        commercial_proposal_gcs_url = await upload_file_to_gcs(
            propostaComercialFile,
            f"propostas/{orgaoSolicitante}_{tituloProjeto}_comercial_{propostaComercialFile.filename}"
        )
        llm_context_data["commercial_proposal_gcs_url"] = commercial_proposal_gcs_url
    else:
        llm_context_data["proposta_comercial_content"] = "Nenhuma proposta comercial PDF fornecida."

    if propostaTecnicaFile and propostaTecnicaFile.filename:
        # Similarmente, extrair texto e passar para o LLM.
        llm_context_data["proposta_tecnica_content"] = await extract_text_from_pdf(propostaTecnicaFile)
        technical_proposal_gcs_url = await upload_file_to_gcs(
            propostaTecnicaFile,
            f"propostas/{orgaoSolicitante}_{tituloProjeto}_tecnica_{propostaTecnicaFile.filename}"
        )
        llm_context_data["technical_proposal_gcs_url"] = technical_proposal_gcs_url
    else:
        llm_context_data["proposta_tecnica_content"] = "Nenhuma proposta técnica PDF fornecida."

    # --- Acrescentar Conhecimento do GCS para o LLM ----
    # Estes são os 'context_data' que o LLM usará para buscar informações e gerar conteúdo.
    # A estrutura de pastas parece ser: "{NomeAcelerador}/BC - {NomeAcelerador}.txt"
    # Ajuste os caminhos conforme a estrutura real do seu bucket.
    
    for product_name in produtosXertica_list:
        # É importante ter os nomes de arquivos padronizados no seu bucket para facilitar
        bc_path = f"{product_name}/BC - {product_name}.txt"
        ds_path = f"{product_name}/DS - {product_name}.txt"
        op_path = f"{product_name}/OP - {product_name}.txt"
        
        # O LLM 'lê' estes conteúdos injetados no prompt
        if get_gcs_file_content(bc_path):
            llm_context_data[f"{product_name}_BC.txt"] = get_gcs_file_content(bc_path)
        if get_gcs_file_content(ds_path):
            llm_context_data[f"{product_name}_DS.txt"] = get_gcs_file_content(ds_path)
        if get_gcs_file_content(op_path):
            llm_context_data[f"{product_name}_OP.txt"] = get_gcs_file_content(op_path)

    # Documentos legais/contratuais de referência
    llm_context_data["MTI_CONTRATO_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MTI/CONTRATO DE PARCERIA 03-2024-MTI - XERTICA - ASSINADO.txt")
    llm_context_data["MPAP_ATA_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MPAP/ATA DE REGISTRO DE PREÇOS Nº 041-2024-XERTICA.txt")
    llm_context_data["RISK_ANALYSIS_CONTEXT.txt"] = get_gcs_file_content("Detecção e Análise de Riscos/Detecção de análise de riscos.txt")

    # --- Chamada ao LLM para Geração de Conteúdo ---
    llm_response = simulate_llm_response(
        "Gerar ETP e TR detalhados com base nas informações fornecidas e no contexto Xertica.ai.",
        llm_context_data
    )
    
    document_subject = llm_response.get("subject", "Documento ETP/TR Gerado pela Xertica.ai")
    etp_content = llm_response.get("etp_content", "Conteúdo do ETP não gerado pelo LLM.")
    tr_content = llm_response.get("tr_content", "Conteúdo do TR não gerado pelo LLM.")

    # --- Autenticação e Construção do Google Docs ---
    docs_service, drive_service = authenticate_google_docs_and_drive()
    if not docs_service or not drive_service:
        raise HTTPException(status_code=500, detail="Serviços do Google Docs/Drive não puderam ser inicializados.")

    try:
        # CRIAÇÃO DO DOCUMENTO UNIFICADO NO GOOGLE DOCS
        new_doc_body = {
            'title': f"{document_subject} - {orgaoSolicitante}",
            'mimeType': 'application/vnd.google-apps.document'
        }
        new_doc_metadata = docs_service.documents().create(body=new_doc_body).execute()
        document_id = new_doc_metadata.get('documentId')
        
        if not document_id:
            raise HTTPException(status_code=500, detail="Falha ao criar novo documento no Google Docs.")
        logger.info(f"Documento único criado: {document_id}")

        # Inserir o conteúdo completo do ETP e TR no documento
        # IMPORTANTE: Considerar usar um TEMPLATE DO GOOGLE DOCS para formatar melhor
        # Ex: substituir placeholders como {{ETP_CONTENT}}, {{TR_CONTENT}}
        # requests = [
        #     {'replaceAllText': {'replaceText': etp_content, 'containsText': {'text': '{{ETP_CONTENT}}'}}},
        #     {'replaceAllText': {'replaceText': tr_content, 'containsText': {'text': '{{TR_CONTENT}}'}}},
        # ] # Se usar template
        
        full_document_text = f"{etp_content}\n\n---\n\n{tr_content}" # Conteúdo direto
        requests = [
            {
                'insertText': {
                    'location': { 'index': 1 }, # Insere no início
                    'text': full_document_text
                }
            }
        ]
        
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
        logger.info(f"Conteúdo ETP/TR inserido no documento {document_id}.")

        # --- Definir Permissão de Leitura Pública e Obter Link ---
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        drive_service.permissions().create(fileId=document_id, body=permission, fields='id').execute()
        logger.info(f"Permissões de leitura pública definidas para {document_id}.")

        file_metadata = drive_service.files().get(fileId=document_id, fields='webViewLink').execute()
        document_link = file_metadata.get('webViewLink')

        if not document_link:
            raise HTTPException(status_code=500, detail="Falha ao obter link compartilhável do Google Docs.")

        logger.info(f"Processo concluído. Documento Google Docs link: {document_link}")
        return JSONResponse(status_code=200, content={
            "success": True, 
            "doc_link": document_link,
            "commercial_proposal_gcs_url": llm_context_data.get("commercial_proposal_gcs_url"),
            "technical_proposal_gcs_url": llm_context_data.get("technical_proposal_gcs_url")
        })

    except HttpError as e:
        error_details = json.loads(e.content.decode()).get('error', {}).get('message', 'Erro desconhecido na API do Google.')
        logger.exception(f"Erro na API do Google Docs/Drive: {error_details}")
        raise HTTPException(status_code=e.resp.status, detail=f"Erro na API do Google Docs/Drive: {error_details}")
    except Exception as e:
        logger.exception(f"Erro inesperado no servidor.")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")

# Para executar localmente para testes (descomente para usar):
# if __name__ == "__main__":
#     import uvicorn
#     # Certifique-se de configurar as variáveis de ambiente em um arquivo .env na raiz do projeto
#     # E/ou configure suas credenciais ADC (Application Default Credentials) localmente via 'gcloud auth application-default login'
#     uvicorn.run(app, host="0.0.0.0", port=8000)
