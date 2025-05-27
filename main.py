import os
import logging
import json
import io # Para lidar com arquivos em memória

from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Google Cloud Imports
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Carrega variáveis de ambiente (para desenvolvimento local) ---
# Este passo só funciona se você tiver um arquivo .env na raiz do seu projeto local.
# No Cloud Run, as variáveis de ambiente são configuradas diretamente no serviço.
load_dotenv()

app = FastAPI(
    title="Gerador de ETP e TR Xertica.ai",
    description="Backend inteligente para gerar documentos ETP e TR com IA da Xertica.ai.",
    version="0.1.0"
)

# --- Configurações CORS ---
# Permite que seu frontend (se estiver em outro domínio) acesse o backend.
# Em produção, substitua "*" pelos domínios específicos do seu frontend para segurança.
origins = ["*"] # Permite qualquer origem para desenvolvimento. CUIDADO em produção!

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variáveis de Ambiente e Inicialização de Clientes GCP / Vertex AI ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.getenv("GCP_PROJECT_LOCATION", "us-central1") # Região para o Vertex AI
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "docsorgaospublicos")

# Validação inicial das variáveis de ambiente obrigatórias
if not GCP_PROJECT_ID:
    logger.error("GCP_PROJECT_ID não está configurado.")
    raise Exception("GCP_PROJECT_ID não configurado. Por favor, defina a variável de ambiente.")

if not GCP_PROJECT_LOCATION:
    logger.error("GCP_PROJECT_LOCATION não está configurado.")
    raise Exception("GCP_PROJECT_LOCATION não configurado. Por favor, defina a variável de ambiente.")

# Inicializa Vertex AI e modelo Gemini
try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_PROJECT_LOCATION)
    # Usando gemini-2.0-flash conforme solicitado
    gemini_model = GenerativeModel("gemini-2.0-flash") 
    _generation_config = GenerationConfig(temperature=0.7, max_output_tokens=8192, response_mime_type="application/json")
    logger.info(f"Vertex AI inicializado com projeto '{GCP_PROJECT_ID}' e localização '{GCP_PROJECT_LOCATION}'. Modelo Gemini-2.0-flash carregado.")
except Exception as e:
    logger.exception(f"Erro ao inicializar Vertex AI ou carregar modelo Gemini: {e}")
    raise HTTPException(status_code=500, detail=f"Falha ao iniciar LLM: {e}. Verifique as permissões da Service Account e as variáveis GCP_PROJECT_ID/GCP_PROJECT_LOCATION.")


# Inicializa cliente de Cloud Storage
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    logger.info("Google Cloud Storage client inicializado.")
except Exception as e:
    logger.exception("Erro ao inicializar Google Cloud Storage client.")
    raise HTTPException(status_code=500, detail=f"Falha ao inicializar GCS: {e}")


# --- Funções Auxiliares para GCP ---

def authenticate_google_docs_and_drive():
    """
    Autentica com as APIs do Google Docs e Drive.
    No Cloud Run, utiliza as credenciais da Service Account do próprio serviço (ADC - Application Default Credentials).
    Certifique-se de que a Service Account tenha as permissões necessárias.
    """
    try:
        docs_service = build('docs', 'v1', cache_discovery=False)
        drive_service = build('drive', 'v3', cache_discovery=False)
        logger.info("Google Docs e Drive services inicializados.")
        return docs_service, drive_service
    except Exception as e:
        logger.exception("Erro ao autenticar/inicializar Google Docs/Drive APIs.")
        raise HTTPException(status_code=500, detail=f"Falha na autenticação da API do Google Docs/Drive: {e}. Verifique as permissões da Service Account (Docs API Editor, Drive API File Creator).")

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
        
        # O .file do UploadFile é um objeto de arquivo (SpoolReader), pode ser lido diretamente.
        # Mas para garantir que o conteúdo seja lido completamente antes do upload, 
        # e evitar problemas com operações assíncronas, lemos para a memória.
        contents = await upload_file.read() 
        upload_file.file.seek(0) # Resetar o ponteiro do arquivo se precisar reler

        blob.upload_from_string(contents, content_type=upload_file.content_type)
        blob.make_public() # Torna o objeto publicamente acessível via URL. CUIDADO com dados sensíveis.
        logger.info(f"Arquivo '{upload_file.filename}' carregado para GCS://{GCS_BUCKET_NAME}/{destination_path} e tornado público.")
        return blob.public_url
    except Exception as e:
        logger.exception(f"Erro ao fazer upload do arquivo '{upload_file.filename}' para GCS.")
        raise HTTPException(status_code=500, detail=f"Falha ao carregar arquivo: {e}")

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """
    Extrai texto de um PDF. Esta implementação é um MOCK/SIMULAÇÃO.
    Para produção, considere usar:
    - `pypdf` ou `pdfminer.six` para PDFs pesquisáveis por texto.
    - Google Cloud Document AI ou Vision AI para PDFs que são imagens ou complexos.
    O uso de `await pdf_file.read()` garante que o conteúdo do arquivo é lido corretamente.
    """
    logger.info(f"Iniciando simulação de extração de texto de PDF: {pdf_file.filename}")
    try:
        # Conteúdo do PDF lido, mas não processado por uma biblioteca de PDF real aqui
        _ = await pdf_file.read() # Lê o conteúdo para que o fluxo do UploadFile seja consumido
        pdf_file.file.seek(0) # Resetar o ponteiro para caso o UploadFile precise ser consumido novamente (ex: para upload GCS)
        
        # Retorna um texto simulado que o Gemini "leria"
        return (f"Conteúdo extraído (SIMULADO) do PDF '{pdf_file.filename}'. "
                f"Este texto representa o que o Gemini 'lê' de sua proposta. "
                f"Contém detalhes sobre os serviços, cronogramas, valores e condições do projeto. "
                f"Exemplificando: A proposta comercial detalha [serviços X, Y, Z], [valor total estimado A], [condições de pagamento B]. "
                f"A proposta técnica aborda [metodologia de implementação C], [equipe D], [integração com sistemas E] e [cronograma detalhado F]."
               )
    except Exception as e:
        logger.exception(f"Erro simulado na extração de texto do PDF {pdf_file.filename}.")
        # Lançar exceção ou retornar uma string de erro caso a extração real falhe.
        return f"Erro na extração de texto do PDF: {e}. Conteúdo indisponível para análise pelo Gemini."

# --- LÓGICA CENTRAL: REAL LLM CALL (GOOGLE GEMINI 2.0 Flash) ---
async def generate_etp_tr_content_with_gemini(llm_context_data: Dict) -> Dict:
    """
    Gera o conteúdo do ETP e TR utilizando o modelo Google Gemini.
    O prompt é construído dinamicamente com base em dados do formulário, GCS e PDFs.
    """
    logger.info(f"Enviando dados para o modelo Gemini para geração de ETP/TR.")

    # Constrói o PROMPT COMPLETO para o Gemini.
    # Este prompt é a INSTRUÇÃO CHAVE que direciona o comportamento do LLM.

    # Dados do Formulário
    orgao_nome = llm_context_data.get('orgaoSolicitante', 'o Órgão Solicitante')
    titulo_projeto = llm_context_data.get('tituloProjeto', 'uma iniciativa')
    justificativa_necessidade = llm_context_data.get('justificativaNecessidade', 'um problema genérico.')
    objetivo_geral = llm_context_data.get('objetivoGeral', 'um objetivo ambicioso.')
    prazos_estimados = llm_context_data.get('prazosEstimados', 'prazos a serem definidos.')
    valor_estimado = llm_context_data.get('valorEstimado')
    modelo_licitacao = llm_context_data.get('modeloLicitacao', 'uma modalidade padrão.')
    parcelamento_contratacao = llm_context_data.get('parcelamentoContratacao', 'Não especificado.')
    justificativa_parcelamento = llm_context_data.get('justificativaParcelamento', 'Não se aplica.')
    contexto_geral_orgao = llm_context_data.get('contextoGeralOrgao', '')


    # Detalhes de Aceleradores Selecionados
    accelerator_details_str = []
    for product_name in llm_context_data.get("produtosXertica", []):
        user_integration_detail = llm_context_data.get(f"integracao_{product_name}", "").strip()
        # Injete aqui os conteúdos reais dos BC/DS/OP para o Gemini ler.
        # Para o prompt do LLM, é melhor ter o conteúdo inline, não apenas uma referência.
        bc_content_prod = llm_context_data.get(f"{product_name}_BC_GCS", "Descrição básica não encontrada.")
        ds_content_prod = llm_context_data.get(f"{product_name}_DS_GCS", "Detalhes técnicos não encontrados.")
        op_content_prod = llm_context_data.get(f"{product_name}_OP_GCS", "Plano operacional não encontrado.")

        detail = f"""
        - **Acelerador:** {product_name}
          - **Resumo Funcional (do GCS):** {bc_content_prod[:500]}... (Primeiros 500 caracteres do Battle Card)
          - **Detalhes Técnicos (do GCS):** {ds_content_prod[:500]}... (Primeiros 500 caracteres do Data Sheet)
          - **Aplicação Específica (do usuário):** {user_integration_detail if user_integration_detail else 'Nenhuma aplicação específica detalhada pelo usuário.'}
        """
        accelerator_details_str.append(detail)
    
    accelerator_details_prompt_section = "\n".join(accelerator_details_str) if accelerator_details_str else "Nenhum acelerador Xertica.ai detalhado."

    # Conteúdo das Propostas PDF (se extraído)
    proposta_comercial_content = llm_context_data.get("proposta_comercial_content", "Não foi possível extrair conteúdo da proposta comercial.")
    proposta_tecnica_content = llm_context_data.get("proposta_tecnica_content", "Não foi possível extrair conteúdo da proposta técnica.")

    # Conteúdo de Contexto Legal/Risco (GCS)
    mti_contrato_exemplo = llm_context_data.get("MTI_CONTRATO_EXEMPLO.txt", "Nenhum contrato MTI de exemplo.")
    mpap_ata_exemplo = llm_context_data.get("MPAP_ATA_EXEMPLO.txt", "Nenhuma ata MPAP de exemplo.")
    risk_analysis_context = llm_context_data.get("RISK_ANALYSIS_CONTEXT.txt", "Nenhum contexto de análise de risco.")


    # INÍCIO DO PROMPT PARA O LLM
    llm_prompt_content = f"""
    Como assistente de IA especializado em licitações públicas e soluções de IA da Xertica.ai, sua missão é elaborar um ETP e um TR rigorosos e completos.

    **Sua resposta FINAL DEVE ser UM OBJETO JSON VÁLIDO.**

    **Formato do JSON de Saída:**
    ```json
    {{
      "subject": "Título do Documento (ETP e TR)",
      "etp_content": "Conteúdo completo do ETP em Markdown",
      "tr_content": "Conteúdo completo do TR em Markdown"
    }}
    ```

    **Instruções Detalhadas para o Conteúdo:**

    **I. Estudo Técnico Preliminar (ETP):**

    *   **Título Principal:** "ESTUDO TÉCNICO PRELIMINAR (ETP)"
    *   **Subtítulos:**
        *   "1. Necessidade da Contratação e Problema a ser Resolvido": Destaque o problema do órgão (`justificativaNecessidade`) e o contextualize, usando `contextoGeralOrgao` se fornecido.
        *   "2. Objetivo da Contratação": Descreva o `objetivoGeral` do órgão de forma clara e mensurável.
        *   "3. Solução Proposta - Aceleradores Xertica.ai":
            *   Descreva cada acelerador selecionado (`produtosXertica`) utilizando os conteúdos fornecidos dos BC/DS/OP do GCS e a aplicação específica (`integracao_`) do usuário.
            *   **Crucial:** Explique como cada acelerador *diretamente* resolve o problema e atinge o objetivo do órgão.
            *   Use o conteúdo da proposta técnica para enriquecer a descrição da solução.
        *   "4. Modelo de Contratação e Justificativa Legal":
            *   Especifique o `modeloLicitacao`.
            *   Justifique legalmente a escolha com base na Lei nº 14.133/2021. Se for dispensa/inexigibilidade (Art. 75, Adesão ATA MPAP, Inexigibilidade ABES), o LLM deve citar os artigos relevantes e usar os **exemplos de contrato MTI, ATA MPAP ou declaração ABES** (se o conteúdo GCS foi fornecido) para embasar a inviabilidade de competição/viabilidade.
        *   "5. Análise Preliminar de Riscos e Mitigação":
            *   Descreva riscos genéricos de projetos de TI/IA no setor público (integração, segurança de dados, adaptação de usuários, cronograma, dependência de fornecedor).
            *   Para cada risco, proponha medidas de mitigação. Se `RISK_ANALYSIS_CONTEXT` foi fornecido, utilize-o como base para os riscos e mitigações.
        *   "6. Cronograma Estimado": Inclua os `prazosEstimados`.
        *   "7. Valor Estimado da Contratação": Mostre o `valorEstimado`. Se `valorEstimado` for N/A, diga "a ser definido em orçamento detalhado."
        *   "8. Aspectos do Parcelamento": Explane sobre a decisão de `parcelamentoContratacao` e adicione a `justificativaParcelamento` se aplicável.

    **II. Termo de Referência (TR):**

    *   **Título Principal:** "TERMO DE REFERÊNCIA (TR)"
    *   **Subtítulos:**
        *   "1. Objeto da Contratação": Reafirme o objeto (o que será contratado) de forma clara e concisa, ligando-o à necessidade e objetivo do órgão.
        *   "2. Especificações Técnicas Detalhadas das Soluções":
            *   Detalhe as funcionalidades e características dos aceleradores selecionados.
            *   Use o conteúdo de BC/DS/OP e, se relevante, a proposta técnica.
            *   Aborde requisitos não funcionais (performance, segurança, escalabilidade) se a informação da proposta técnica/acelerador permitir.
        *   "3. Regime de Execução": Especifique `modeloLicitacao`.
        *   "4. Condições e Prazos de Execução": Detalhe `prazosEstimados`.
        *   "5. Anexos":
            *   Mencione a Proposta Comercial e Proposta Técnica como anexos, indicando que foram fornecidas como PDFs.

    **III. Qualidade e Formato:**
    *   O tom deve ser formal, técnico e alinhado aos padrões de documentos governamentais brasileiros.
    *   Use Markdown para estruturar o conteúdo (títulos com #, listas com -, negrito com **, etc.).
    *   Sua resposta DEVE ser um JSON válido contendo as chaves `subject`, `etp_content`, `tr_content`.

    ---
    **DADOS FORNECIDOS PELO USUÁRIO (Órgão Solicitante):**
    {json.dumps(llm_context_data, indent=2)}

    ---
    **CONTEÚDO EXTRAÍDO DAS PROPOSTAS XERTICA.AI (Anexos PDF):**
    Proposta Comercial: {proposta_comercial_content}
    Proposta Técnica: {proposta_tecnica_content}

    ---
    **CONTEÚDO DE CONTEXTO GCS (Battle Cards, Data Sheets, OP, Documentos Legais):**
    {gcs_accel_str}
    {gcs_legal_str}

    ---
    Agora, gere o JSON com o 'subject', 'etp_content' e 'tr_content' conforme as instruções.
    """

    # --- 4. CHAMADA REAL AO MODELO GEMINI ---
    try:
        response = await gemini_model.generate_content_async(
            llm_prompt_content,
            generation_config=_generation_config
        )
        
        # O modelo é instruído a retornar JSON diretamente via response_mime_type no generation_config.
        # Precisamos ter certeza que o conteúdo de texto da resposta é um JSON válido.
        if response.text:
            generated_content = json.loads(response.text)
            logger.info("Conteúdo gerado pelo Gemini e parseado com sucesso.")
            return generated_content
        else:
            logger.warning("Resposta do Gemini vazia ou não contém texto.")
            raise Exception("Resposta vazia do modelo Gemini.")

    except json.JSONDecodeError as e:
        logger.exception("Erro ao parsear JSON da resposta do Gemini.")
        # É crucial logar a resposta bruta do Gemini aqui para depuração.
        # response.text pode ser acessível aqui.
        raise HTTPException(status_code=500, detail=f"Erro no formato JSON retornado pelo Gemini: {e}")
    except Exception as e:
        logger.exception(f"Erro ao chamar a API do Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na geração de conteúdo via IA: {e}")


# --- Endpoint Principal da API ---

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
    logger.info(f"Requisição para '{tituloProjeto}' de '{orgaoSolicitante}' recebida.")

    # 1. Coletar e Unificar Dados do Formulário
    form_data = await request.form()
    produtosXertica_list = form_data.getlist("produtosXertica") # Obtém a lista de produtos selecionados

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

    # Adicionar os campos dinâmicos "integracao_[produto]" do formulário
    for key in form_data.keys():
        if key.startswith("integracao_"):
            llm_context_data[key] = form_data.get(key)
    
    # 2. Processar e Unificar Conteúdo de Propostas PDF e Upload no GCS
    # Extrair texto do PDF e salvar o arquivo no GCS (retornando URL).
    # O texto extraído será usado pelo LLM.
    if propostaComercialFile and propostaComercialFile.filename:
        llm_context_data["proposta_comercial_content"] = await extract_text_from_pdf(propostaComercialFile)
        llm_context_data["commercial_proposal_gcs_url"] = await upload_file_to_gcs(
            propostaComercialFile,
            f"propostas/{orgaoSolicitante}_{tituloProjeto}_comercial_{propostaComercialFile.filename}"
        )
    else:
        llm_context_data["proposta_comercial_content"] = "Nenhuma proposta comercial PDF fornecida."
        llm_context_data["commercial_proposal_gcs_url"] = None

    if propostaTecnicaFile and propostaTecnicaFile.filename:
        llm_context_data["proposta_tecnica_content"] = await extract_text_from_pdf(propostaTecnicaFile)
        llm_context_data["technical_proposal_gcs_url"] = await upload_file_to_gcs(
            propostaTecnicaFile,
            f"propostas/{orgaoSolicitante}_{tituloProjeto}_tecnica_{propostaTecnicaFile.filename}"
        )
    else:
        llm_context_data["proposta_tecnica_content"] = "Nenhuma proposta técnica PDF fornecida."
        llm_context_data["technical_proposal_gcs_url"] = None

    # 3. Acrescentar Conhecimento do GCS para o LLM
    # Buscar e injetar conteúdos de arquivos TXT de Battle Cards, Data Sheets, Operational Plans
    # e documentos legais/contextuais.
    
    # Adicionar conteúdo de aceleradores Xertica do GCS
    for product_name in produtosXertica_list:
        bc_path = f"{product_name}/BC - {product_name}.txt"
        ds_path = f"{product_name}/DS - {product_name}.txt"
        op_path = f"{product_name}/OP - {product_name}.txt"
        
        # Carrega o conteúdo e o adiciona ao contexto do LLM
        if get_gcs_file_content(bc_path): llm_context_data[f"{product_name}_BC_GCS"] = get_gcs_file_content(bc_path)
        if get_gcs_file_content(ds_path): llm_context_data[f"{product_name}_DS_GCS"] = get_gcs_file_content(ds_path)
        if get_gcs_file_content(op_path): llm_context_data[f"{product_name}_OP_GCS"] = get_gcs_file_content(op_path)

    # Adicionar documentos legais/contratuais de referência do GCS
    llm_context_data["MTI_CONTRATO_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MTI/CONTRATO DE PARCERIA 03-2024-MTI - XERTICA - ASSINADO.txt")
    llm_context_data["MPAP_ATA_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MPAP/ATA DE REGISTRO DE PREÇOS Nº 041-2024-XERTICA.txt")
    llm_context_data["RISK_ANALYSIS_CONTEXT.txt"] = get_gcs_file_content("Detecção e Análise de Riscos/Detecção de análise de riscos.txt")

    # 4. Chamar o LLM para Geração de Conteúdo
    llm_response = await generate_etp_tr_content_with_gemini(llm_context_data)
    
    document_subject = llm_response.get("subject", f"Documento ETP/TR: {orgaoSolicitante} - {tituloProjeto}")
    etp_content = llm_response.get("etp_content", "Conteúdo do ETP não gerado pelo LLM ou com erro.")
    tr_content = llm_response.get("tr_content", "Conteúdo do TR não gerado pelo LLM ou com erro.")

    # 5. Autenticação e Criação do Google Docs
    docs_service, drive_service = authenticate_google_docs_and_drive()

    try:
        # CRIAÇÃO DO DOCUMENTO UNIFICADO NO GOOGLE DOCS
        new_doc_body = {
            'title': document_subject,
            'mimeType': 'application/vnd.google-apps.document'
        }
        new_doc_metadata = docs_service.documents().create(body=new_doc_body).execute()
        document_id = new_doc_metadata.get('documentId')
        
        if not document_id:
            raise HTTPException(status_code=500, detail="Falha ao criar novo documento no Google Docs.")
        logger.info(f"Documento único criado: {document_id}")

        # Inserir o conteúdo completo do ETP e TR no documento.
        # Conteúdo em Markdown será inserido como texto puro.
        full_document_text = f"{etp_content}\n\n---\n\n{tr_content}" 
        requests = [
            {
                'insertText': {
                    'location': { 'index': 1 }, # No início do documento
                    'text': full_document_text
                }
            }
        ]
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
        logger.info(f"Conteúdo ETP/TR inserido no documento {document_id}.")

        # 6. Definir Permissão de Leitura Pública e Obter Link
        permission = {
            'type': 'anyone', # Acesso público
            'role': 'reader'  # Permissão de leitura
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
        logger.exception(f"Erro na geração do documento.")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")

# Para executar localmente para testes (descomente para usar):
# if __name__ == "__main__":
#     import uvicorn
#     # Configure as variáveis de ambiente em um arquivo .env na raiz do projeto
#     # Ex: GCP_PROJECT_ID="seu-projeto-gcp", GCP_PROJECT_LOCATION="us-central1", GCS_BUCKET_NAME="seu-bucket-gcs"
#     # Ou configure suas credenciais ADC localmente via 'gcloud auth application-default login'
#     uvicorn.run(app, host="0.0.0.0", port=8000)
