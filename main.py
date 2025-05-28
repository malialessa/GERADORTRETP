import os
import logging
import json
import io 
from datetime import date 

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
load_dotenv()

app = FastAPI(
    title="Gerador de ETP e TR Xertica.ai",
    description="Backend inteligente para gerar documentos ETP e TR com IA da Xertica.ai.",
    version="0.1.0"
)

# --- Configurações CORS ---
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
    **FUNÇÃO MOCK PARA EXTRAÇÃO RÚSTICA DE TEXTO DE PDF.**
    EM UMA IMPLEMENTAÇÃO REAL E ROBUSTA DE PRODUÇÃO, ESPECIALMENTE PARA DOCUMENTOS COMPLEXOS:
    - **Usaria Google Cloud Document AI (Recomendado):** Para alta precisão, reconhecimento de campos, tabelas, etc.
    - **OU Google Cloud Vision AI:** Para OCR em PDFs escaneados ou baseados em imagem.
    - **OU Bibliotecas Python robustas como `pypdf`, `pdfminer.six` (para PDFs text-searchable).
    """
    logger.info(f"Simulando extração de texto de PDF: {pdf_file.filename}")
    try:
        _ = await pdf_file.read() 
        pdf_file.file.seek(0)
        
        return (f"Conteúdo extraído (SIMULADO) do PDF '{pdf_file.filename}'. "
                f"Este texto representa o que o Gemini 'lê' de sua proposta. "
                f"Contém detalhes sobre os serviços, cronogramas, valores e condições do projeto. "
                f"Exemplificando: A proposta comercial detalha [serviços X, Y, Z], [valor total estimado A], [condições de pagamento B]. "
                f"A proposta técnica aborda [metodologia de implementação C], [equipe D], [integração com sistemas E] e [cronograma detalhado F]."
               )
    except Exception as e:
        logger.exception(f"Erro simulado na extração de texto do PDF {pdf_file.filename}.")
        return f"Erro na extração de texto do PDF: {e}. Conteúdo indisponível para análise pelo Gemini."

# --- LÓGICA CENTRAL: REAL LLM CALL (GOOGLE GEMINI 2.0 Flash) ---
async def generate_etp_tr_content_with_gemini(llm_context_data: Dict) -> Dict:
    """
    Gera o conteúdo do ETP e TR utilizando o modelo Google Gemini.
    O prompt é construído dinamicamente com base em dados do formulário, GCS e PDFs.
    """
    logger.info(f"Iniciando chamada ao modelo Gemini para geração de ETP/TR.")

    # --- Construção dos conteúdos GCS formatados para o PROMPT (dentro do escopo da função) ---
    gcs_accel_str = "\n".join(f"Conteúdo GCS - {k}:\n{v}\n" for k, v in llm_context_data.get('gcs_accelerator_content', {}).items() if v)
    gcs_legal_str = "\n".join(f"Conteúdo GCS - {k}:\n{v}\n" for k, v in llm_context_data.get('gcs_legal_context_content', {}).items() if v)

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
    
    # Data para o ETP/TR (Formato: dia de mês_por_extenso de ano)
    today = date.today()
    meses_pt = { # Mês por extenso para formato de data
        1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho",
        7: "julho", 8: "agosto", 9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"
    }
    mes_extenso = meses_pt[today.month]
    
    # Inferir esfera administrativa
    esfera_administrativa = "Federal" # Default
    if "municipal" in orgao_nome.lower() or "pref." in orgao_nome.lower() or "prefeitura" in orgao_nome.lower():
        esfera_administrativa = "Municipal"
    elif "estadual" in orgao_nome.lower() or "governo do estado" in orgao_nome.lower() or "secretaria de estado" in orgao_nome.lower() or "tj" in orgao_nome.lower():
        esfera_administrativa = "Estadual"
    
    # Nome da localidade para o ETP/TR, inferindo cidade/estado do orgao_nome
    # Isso é uma heurística, idealmente o formulário teria campos para Cidade/UF.
    partes_nome_orgao = orgao_nome.split()
    local_etp_cidade = partes_nome_orgao[0] if len(partes_nome_orgao) > 0 else "[Cidade]"
    local_etp_uf = ""
    for parte in partes_nome_orgao:
        if len(parte) == 2 and parte.isupper(): # Tenta encontrar uma UF (2 letras maiúsculas)
            local_etp_uf = parte
            break
    if not local_etp_uf: local_etp_uf = "[UF]"
    
    local_etp_full = f"{local_etp_cidade} ({local_etp_uf})"


    # Detalhes de Aceleradores Selecionados para o LLM
    accelerator_details_prompt_list = []
    for product_name in llm_context_data.get("produtosXertica", []):
        user_integration_detail = llm_context_data.get(f"integracao_{product_name}", "").strip()
        bc_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_BC_GCS", "Dados do Battle Card não disponíveis.")
        ds_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_DS_GCS", "Dados do Data Sheet não disponíveis.")
        op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_OP_GCS", "Dados do Operational Plan não disponíveis.")

        accelerator_details_prompt_list.append(f"""
        - **Acelerador:** {product_name}
          - **Resumo do Battle Card (GCS):** {bc_content_prod_raw[:1000]}...
          - **Detalhes do Data Sheet (GCS):** {ds_content_prod_raw[:1000]}...
          - **Detalhes do Operational Plan (GCS):** {op_content_prod_raw[:1000]}...
          - **Aplicação Específica no Órgão (Input do Usuário):** {user_integration_detail if user_integration_detail else 'Não detalhado pelo usuário, o LLM deve inferir com base no problema/solução.'}
        """)
    
    accelerator_details_prompt_section = "\n".join(accelerator_details_prompt_list) if accelerator_details_prompt_list else "Nenhum acelerador Xertica.ai selecionado."

    # Conteúdo das Propostas PDF (já extraído para texto)
    proposta_comercial_content = llm_context_data.get("proposta_comercial_content", "Não foi possível extrair conteúdo da proposta comercial.")
    proposta_tecnica_content = llm_context_data.get("proposta_tecnica_content", "Não foi possível extrair conteúdo da proposta técnica.")

    # Mapa de Preços (APENAS CABEÇALHO - CONTEÚDO REMOVIDO CONFORME SOLICITADO)
    # A estrutura está aqui para o LLM entender o formato esperado para um mapa de preços.
    # O LLM é instruído a racionalizar os valores, citando este formato.
    price_map_federal_template = """
Tipo de Licença | Fonte (Contrato) | Valor Unitário Anual (R$) | Valor Mensal (R$) | Qtd. Ref. | Valor Total Estimado (R$) Anual
---|---|---|---|---|---
"""[1:] 

    price_map_municipal_template = """
Tipo de Licença | Fonte (Contrato) | Empresa Contratada | Valor Unitário Anual (R$) | Valor Mensal (R$) | Qtd. Ref. | Valor Total Estimado (R$) Anual
---|---|---|---|---|---|---
"""[1:]
    
    price_map_to_use_template = ""
    if esfera_administrativa == "Federal":
        price_map_to_use_template = price_map_federal_template
    else: # Estadual ou Municipal usa o mesmo modelo do municipal para agora
        price_map_to_use_template = price_map_municipal_template

    # INÍCIO DO PROMPT PARA O LLM
    llm_prompt_content = f"""
    Contexto da Solicitação:
    Você é um assistente de IA altamente especializado em elaboração de documentos técnicos e legais para o setor público brasileiro (esferas Federal, Estadual e Municipal), com expertise em licitações (Lei nº 14.133/2021, Decreto Estadual 21.872/2023 [Piauí, se relevante], e Lei 13.303/2016 para empresas estatais), e nas soluções de Inteligência Artificial da Xertica.ai.

    Sua tarefa é gerar duas seções completas (ETP e TR), bem fundamentadas e **adaptadas à esfera administrativa do órgão solicitante ({esfera_administrativa})**, utilizando o Markdown para formatação. Você deve preencher todos os `{{placeholders}}` nos modelos fornecidos e gerar todo o conteúdo dinâmico.

    **Sua resposta FINAL DEVE ser UM OBJETO JSON VÁLIDO.**

    **Formato do JSON de Saída:**
    ```json
    {{
      "subject": "Título Descritivo do Documento (ETP e TR)",
      "etp_content": "Conteúdo COMPLETO do ETP em Markdown, preenchendo todos os `{{placeholders}}` e gerando texto necessário.",
      "tr_content": "Conteúdo COMPLETO do TR em Markdown, preenchendo todos os `{{placeholders}}` e gerando texto necessário."
    }}
    ```

    **Regras Detalhadas para Geração de Conteúdo:**

    1.  **Adaptação Linguística e Legal Específica por Esfera:** Adapte toda a linguagem e referências legais (por exemplo, "Lei Orçamentária Anual da União" vs. "Lei Orçamentária Anual do Estado/Município") com base na esfera administrativa **{esfera_administrativa}** do `orgaoSolicitante`.
    2.  **Siga os Modelos Rigorosamente:** Preencha os Markdowns para ETP e TR. **Importante:** Não inclua explicações ou comentários do modelo original (texto em vermelho, verde, azul, ou anotações `< Identificar... >` como parte da saída final do documento).
    3.  **Racionalização e Preenchimento de Placeholders:** Para campos não preenchidos pelo usuário (como `numero_processo_administrativo`, `CNAE`, nome dos responsáveis), o LLM deve:
        *   Preencher com um placeholder claro como `[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]`.
        *   Gerar um exemplo genérico e consistente para os campos que podem ter um valor padrão (ex: `Processo Administrativo nº XXXXXX/ANO`).
        *   Para **CNAE**, sugira um CNAE de TI comum (ex: "6204-0/00 - Consultoria em tecnologia da informação") e ADICIONE TEXTO explicitando que "o CNAE específico deverá ser confirmado e preenchido pelo órgão".
        *   Para prazos de recebimento e faturamento/pagamento, utilize valores razoáveis e comuns para contratos públicos (ex: "5 (cinco) dias úteis", "15 (quinze) dias corridos").
    4.  **Integração do Mapa de Preços:** Use o "MAPA DE PREÇOS DE REFERÊNCIA" (cuja estrutura está abaixo) para justificar a estimativa de custo. Se o `valorEstimado` do formulário foi N/A, rationalize-o com base em uma suposta pesquisa de mercado ou em faixas razoáveis, mencionando que "o valor está em consonância com dados de mercado e propostas anteriores", **referenciando a estrutura do mapa de preços como base dessa análise**. O LLM deve inferir valores plausíveis e justificá-los mesmo sem dados completos na tabela.
    5.  **Conexão Inteligente (RAG):** Use as informações dos Battle Cards, Data Sheets, Operational Plans (do GCS) e as propostas anexadas (PDFs) para descrever as funcionalidades dos aceleradores e como eles **diretamente resolvem** a `justificativaNecessidade` e **atingem** o `objetivoGeral` do órgão.
    6.  **Justificativa Legal Robusta:** Para o `modeloLicitacao`, use o conhecimento da Lei 14.133/2021 e da Lei 13.303/2016 e o contexto legal do GCS (MTI, MPAP, SERPRO MoU, ABES, Riscos) para fornecer justificativas legais detalhadas, citando artigos relevantes e explicando a aplicabilidade.
    7.  **Outros Detalhes:**
        *   Use `local_etp_full` (ex: "Cuiabá (MT)") e a data de hoje para os cabeçalhos/rodapés.
        *   O título do ETP/TR deve ser claro e profissional.

    ---
    **DADOS FORNECIDOS PELO USUÁRIO (Órgão Solicitante):**
    {json.dumps(llm_context_data, indent=2)}

    ---
    **CONTEÚDO EXTRAÍDO DAS PROPOSTAS XERTICA.AI (Anexos PDF):**
    Proposta Comercial: {proposta_comercial_content}
    Proposta Técnica: {proposta_tecnica_content}

    ---
    **MAPA DE PREÇOS DE REFERÊNCIA PARA CONTRATAÇÃO (Estrutura Fornecida para Orientação do LLM):**
    Utilize esta estrutura para fundamentar a seção de estimativa de preço, mesmo que o conteúdo esteja vazio.
    {price_map_to_use_template}

    ---
    **CONTEÚDO DE CONTEXTO GCS (Battle Cards, Data Sheets, OP, Documentos Legais):**
    {gcs_accel_str}
    {gcs_legal_str}

    ---
    Gere o objeto JSON agora, seguindo todas as instruções e o formato especificado.
    """

    try:
        response = await gemini_model.generate_content_async(
            llm_prompt_content,
            generation_config=_generation_config
        )
        
        # Log da resposta RAW do Gemini para depuração
        logger.info(f"Resposta RAW do Gemini: {response.text[:2000]}...")
        
        if response.text:
            generated_content = json.loads(response.text)
            logger.info("Resposta do Gemini recebida e parseada como JSON.")
            return generated_content
        else:
            logger.warning("Resposta do Gemini vazia ou não contém texto.")
            raise Exception("Resposta vazia do modelo Gemini.")

    except json.JSONDecodeError as e:
        logger.error(f"Erro ao parsear JSON da resposta do Gemini: {e}.")
        logger.error(f"Resposta RAW do Gemini que causou o erro: {response.text}") 
        raise HTTPException(status_code=500, detail=f"Erro no formato JSON retornado pelo Gemini: {e}. Resposta RAW logada.")
    except Exception as e:
        logger.exception(f"Erro ao chamar a API do Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na geração de conteúdo via IA: {e}")


# --- Endpoint Principal da API ---

@app.post("/generate_etp_tr")
async def generate_etp_tr_endpoint(
    request: Request, 
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

    # Adicionar os campos dinâmicos "integracao_[produto]" do formulário
    for key in form_data.keys():
        if key.startswith("integracao_"):
            llm_context_data[key] = form_data.get(key)
    
    # 2. Processar e Unificar Conteúdo de Propostas PDF e Upload no GCS
    if propostaComercialFile and propostaComercialFile.filename:
        logger.info(f"Detectada Proposta Comercial: {propostaComercialFile.filename}")
        llm_context_data["proposta_comercial_content"] = await extract_text_from_pdf(propostaComercialFile)
        llm_context_data["commercial_proposal_gcs_url"] = await upload_file_to_gcs(
            propostaComercialFile,
            f"propostas/{orgaoSolicitante}_{tituloProjeto}_comercial_{propostaComercialFile.filename}"
        )
    else:
        llm_context_data["proposta_comercial_content"] = "Nenhuma proposta comercial PDF fornecida."
        llm_context_data["commercial_proposal_gcs_url"] = None

    if propostaTecnicaFile and propostaTecnicaFile.filename:
        logger.info(f"Detectada Proposta Técnica: {propostaTecnicaFile.filename}")
        llm_context_data["proposta_tecnica_content"] = await extract_text_from_pdf(propostaTecnicaFile)
        llm_context_data["technical_proposal_gcs_url"] = await upload_file_to_gcs(
            propostaTecnicaFile,
            f"propostas/{orgaoSolicitante}_{tituloProjeto}_tecnica_{propostaTecnicaFile.filename}"
        )
    else:
        llm_context_data["proposta_tecnica_content"] = "Nenhuma proposta técnica PDF fornecida."
        llm_context_data["technical_proposal_gcs_url"] = None

    # 3. Acrescentar Conhecimento do GCS para o LLM
    llm_context_data['gcs_accelerator_content'] = {}
    llm_context_data['gcs_legal_context_content'] = {}

    # Adicionar conteúdo de aceleradores Xertica do GCS
    for product_name in produtosXertica_list:
        bc_path = f"{product_name}/BC - {product_name}.txt"
        ds_path = f"{product_name}/DS - {product_name}.txt"
        op_path = f"{product_name}/OP - {product_name}.txt"
        
        if get_gcs_file_content(bc_path): llm_context_data['gcs_accelerator_content'][f"{product_name}_BC_GCS"] = get_gcs_file_content(bc_path)
        if get_gcs_file_content(ds_path): llm_context_data['gcs_accelerator_content'][f"{product_name}_DS_GCS"] = get_gcs_file_content(ds_path)
        if get_gcs_file_content(op_path): llm_context_data['gcs_accelerator_content'][f"{product_name}_OP_GCS"] = get_gcs_file_content(op_path)

    # Adicionar documentos legais/contratuais de referência do GCS
    llm_context_data['gcs_legal_context_content']["MTI_CONTRATO_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MTI/CONTRATO DE PARCERIA 03-2024-MTI - XERTICA - ASSINADO.txt")
    llm_context_data['gcs_legal_context_content']["MPAP_ATA_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MPAP/ATA DE REGISTRO DE PREÇOS Nº 041-2024-XERTICA.txt")
    llm_context_data['gcs_legal_context_content']["RISK_ANALYSIS_CONTEXT.txt"] = get_gcs_file_content("Detecção e Análise de Riscos/Detecção de análise de riscos.txt")
    llm_context_data['gcs_legal_context_content']["SERPRO_MOU_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/Serpro/[Xertica & Serpro] Memorando de Entendimento (MoU) - VersãoFinal.txt") 

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
        full_document_text = f"{etp_content}\n\n---\n\n{tr_content}" 
        requests_docs_api = [ 
            {
                'insertText': {
                    'location': { 'index': 1 }, 
                    'text': full_document_text
                }
            }
        ]
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests_docs_api}).execute()
        logger.info(f"Conteúdo ETP/TR inserido no documento {document_id}.")

        # 6. Definir Permissão de Leitura Pública e Obter Link
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
        logger.exception(f"Erro na geração do documento.")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")

# Para executar localmente para testes (descomente para usar):
# if __name__ == "__main__":
#     import uvicorn
#     # Configure as variáveis de ambiente em um arquivo .env na raiz do projeto
#     # Ex: GCP_PROJECT_ID="seu-projeto-gcp", GCP_PROJECT_LOCATION="us-central1", GCS_BUCKET_NAME="seu-bucket-gcs"
#     # Ou configure suas credenciais ADC localmente via 'gcloud auth application-default-login'
#     # (Instale google-auth-oauthlib para isso)
#     uvicorn.run(app, host="0.0.0.0", port=8000)
