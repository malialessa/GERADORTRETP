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
GCP_PROJECT_LOCATION = os.getenv("GCP_PROJECT_LOCATION", "us-central1") 
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
    gemini_model = GenerativeModel("gemini-1.5-flash-001") # Modelo "gemini-1.5-flash-001" é mais novo e pode ser melhor para JSON
    _generation_config = GenerationConfig(temperature=0.7, max_output_tokens=8192, response_mime_type="application/json")
    logger.info(f"Vertex AI inicializado com projeto '{GCP_PROJECT_ID}' e localização '{GCP_PROJECT_LOCATION}'. Modelo Gemini-1.5-flash-001 carregado.")
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


def apply_basic_markdown_to_docs_requests(markdown_content: str) -> List[Dict]:
    """
    Converte um subconjunto de Markdown para uma lista de requests da Google Docs API.
    ATENÇÃO: Esta é uma implementação BÁSICA e não um parser de Markdown completo.
    Lida com: # (Heading 1), ## (Heading 2), **bold**, - (list items), e parágrafos normais.
    Ignora tabelas e outros elementos complexos por enquanto.
    """
    requests = []
    lines = markdown_content.split('\n')
    
    # Track the last inserted position
    current_index = 1 

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
            current_index += 1
            continue

        # Headings
        if line_stripped.startswith('## '):
            level = 2
            text = line_stripped[3:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            requests.append({"updateParagraphStyle": {"range": {"startIndex": current_index, "endIndex": current_index + len(text) -1 }, "paragraphStyle": {"namedStyleType": f"HEADING_{level}"}}})
            current_index += len(text) + 1
        elif line_stripped.startswith('# '):
            level = 1
            text = line_stripped[2:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": text + "\n"}})
            requests.append({"updateParagraphStyle": {"range": {"startIndex": current_index, "endIndex": current_index + len(text) -1 }, "paragraphStyle": {"namedStyleType": f"HEADING_{level}"}}})
            current_index += len(text) + 1
        
        # Lists (simple dash list)
        elif line_stripped.startswith('- '):
            text = line_stripped[2:]
            requests.append({"insertText": {"location": {"index": current_index}, "text": "• " + text + "\n"}})
            requests.append({"createParagraphBullets": {"range": {"startIndex": current_index, "endIndex": current_index + len(text) + 2}, "bulletPreset": "BULLET_DISC_CIRCLE"}})
            current_index += len(text) + 3 # +2 for bullet and space, +1 for newline
        
        # Bold
        elif "**" in line_stripped:
            parts = []
            last_end = 0
            for match in re.finditer(r'\*\*(.*?)\*\*', line_stripped):
                start, end = match.span()
                bold_text = match.group(1)
                
                # Add text before bold
                if start > last_end:
                    requests.append({"insertText": {"location": {"index": current_index}, "text": line_stripped[last_end:start]}})
                    current_index += len(line_stripped[last_end:start])
                
                # Add bold text
                requests.append({"insertText": {"location": {"index": current_index}, "text": bold_text}})
                requests.append({"updateTextStyle": {"range": {"startIndex": current_index, "endIndex": current_index + len(bold_text)}, "textStyle": {"bold": True}}})
                current_index += len(bold_text)
                last_end = end
                
            # Add remaining text after last bold
            if last_end < len(line_stripped):
                requests.append({"insertText": {"location": {"index": current_index}, "text": line_stripped[last_end:]}})
                current_index += len(line_stripped[last_end:])
            requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
            current_index += 1 # for newline

        # Normal Paragraph
        else:
            requests.append({"insertText": {"location": {"index": current_index}, "text": line_stripped + "\n"}})
            current_index += len(line_stripped) + 1 # +1 for newline
            
    return requests

# --- LÓGICA CENTRAL: REAL LLM CALL (GOOGLE GEMINI 1.5 Flash) ---
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
    local_etp_full = f"[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO], {mes_extenso} de {today.year}"


    # Detalhes de Aceleradores Selecionados para o LLM
    accelerator_details_prompt_list = []
    for product_name in llm_context_data.get("produtosXertica", []):
        user_integration_detail = llm_context_data.get(f"integracao_{product_name}", "").strip()
        # O LLM que vai pegar deste BC_GCS por exemplo, e escrever a descrição
        bc_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_BC_GCS", "Dados do Battle Card não disponíveis.")
        ds_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_DS_GCS", "Dados do Data Sheet não disponíveis.")
        op_content_prod_raw = llm_context_data.get('gcs_accelerator_content', {}).get(f"{product_name}_OP_GCS", "Dados do Operational Plan não disponíveis.")

        accelerator_details_prompt_list.append(f"""
        - **Acelerador:** {product_name}
          - **Resumo do Battle Card (GCS):** {bc_content_prod_raw[:min(1000, len(bc_content_prod_raw))]}...
          - **Detalhes do Data Sheet (GCS):** {ds_content_prod_raw[:min(1000, len(ds_content_prod_raw))]}...
          - **Detalhes do Operational Plan (GCS):** {op_content_prod_raw[:min(1000, len(op_content_prod_raw))]}...
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

    Sua tarefa é gerar duas seções completas (ETP e TR) em Markdown, **seguindo rigorosamente os modelos de estrutura ETP e TR (que serão fornecidos abaixo) e adaptando o conteúdo à esfera administrativa do órgão solicitante ({esfera_administrativa})**. Você deve preencher todos os `{{placeholders}}` nos modelos e gerar todo o conteúdo dinâmico.

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

    1.  **Adaptação Linguística e Legal Específica por Esfera:** Adapte toda a linguagem e referências legais (por exemplo, "Lei Orçamentária Anual da União" vs. "Lei Orçamentária Anual do Estado/Município", "MINISTÉRIO PÚBLICO da {NÍVEL ADMINISTRATIVO}") com base na esfera administrativa **{esfera_administrativa}** do `orgaoSolicitante`.
    2.  **Racionalize Informações Incompletas:** Para campos que "deverão ser preenchidos pelo órgão" e que não estão na entrada (como número de processo, nomes de responsáveis técnicos), o LLM deve:
        *   Preencher com um placeholder claro em **negrito** e entre `[ ]` como `**[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**`.
        *   No caso de número de processo (`Processo Administrativo nº`), gerar um formato genérico e consistente (`XXXXXX/ANO`).
        *   Para **CNAE**, sugira um CNAE de TI comum (ex: "6204-0/00 - Consultoria em tecnologia da informação") e ADICIONE TEXTO explicitando que "o CNAE específico deverá ser confirmado e preenchido pelo órgão".
        *   Para prazos de recebimento e faturamento/pagamento, utilize valores razoáveis e comuns para contratos públicos (ex: "5 (cinco) dias úteis", "15 (quinze) dias corridos").
    3.  **Integração do Mapa de Preços:** Use o "MAPA DE PREÇOS DE REFERÊNCIA" (cuja estrutura está abaixo) para justificar a estimativa de custo. Se o `valorEstimado` do formulário foi N/A, rationalize-o com base em uma suposta pesquisa de mercado ou em faixas razoáveis, mencionando que "o valor está em consonância com dados de mercado e propostas anteriores", **referenciando a estrutura do mapa de preços como base dessa análise**. O LLM deve inferir valores plausíveis e justificá-los mesmo sem dados completos na tabela.
    4.  **Conexão Inteligente (RAG - Deep Research):**
        *   Utilize as informações dos Battle Cards, Data Sheets, Operational Plans (do GCS) e as propostas anexadas (PDFs) para realizar um "deep research" e descrever as funcionalidades dos aceleradores e como eles **diretamente resolvem** a `justificativaNecessidade` e **atingem** o `objetivoGeral` do órgão.
        *   **CRÍTICO:** Nas seções de "Descrição da Solução", "Requisitos", "Levantamento de Mercado" e "Justificativa da Solução", o LLM deve detalhar **o que a Xertica.ai faz de ÚNICO ou MELHOR** (ex: integração nativa com GCP, especialização no setor público, notória especialização, vivência em projetos similares como o da PROCERGS/MPRS/TJES) em comparação com alternativas, usando o contexto do GCS e PDFs.

    5.  **Justificativa Legal Robusta:** Para o `modeloLicitacao`, use o conhecimento da Lei 14.133/2021 e da Lei 13.303/2016 e o contexto legal do GCS (MTI, MPAP, SERPRO MoU, ABES, Riscos) para fornecer justificativas legais detalhadas, citando artigos relevantes e explicando a aplicabilidade.

    6.  **Formatação Markdown:** A saída deve ser Markdown valido. Use `#` para H1, `##` para H2, `###` para H3, `*` para negrito, `-` para listas. **NÃO INCLUA TEXTOS DE INSTRUÇÃO DO MODELO ORIGINAL (VERMELHO, VERDE, AZUL, COMENTÁRIOS).**

    ---
    **DADOS FORNECIDOS PELO USUÁRIO (Órgão Solicitante):**
    ```json
    {json.dumps(llm_context_data, indent=2)}
    ```

    ---
    **CONTEÚDO EXTRAÍDO DAS PROPOSTAS XERTICA.AI (Anexos PDF):**
    Proposta Comercial: {proposta_comercial_content}
    Proposta Técnica: {proposta_tecnica_content}

    ---
    **MAPA DE PREÇOS DE REFERÊNCIA PARA CONTRATAÇÃO (Estrutura Fornecida para Orientação do LLM):**
    Utilize esta estrutura para fundamentar a seção de estimativa de preço, mesmo que o conteúdo esteja vazio.
    ```
    {price_map_to_use_template}
    ```

    ---
    **CONTEÚDO DE CONTEXTO GCS (Battle Cards, Data Sheets, OP, Documentos Legais):**
    {gcs_accel_str}
    {gcs_legal_str}

    ---
    **Mapeamento de Placeholders para Preenchimento (Para o LLM:**
    *   `{{acelerador_list_summary}}`: Resumo dos aceleradores selecionados.
    *   `{{numero_processo_administrativo}}`: Processo administrativo (genérico ou do contexto, ex: XXXXXX/ANO).
    *   `{{local_etp}}`: Onde `local_etp_full` foi definido ("Cidade (UF)").
    *   `{{mes_etp}}`: Mês atual.
    *   `{{ano_etp}}`: Ano atual.
    *   `{{introducao_contexto}}: Detalha o texto de introdução do ETP.
    *   `{{introducao_referencia_legal_etp}}`: Adiciona a referência legal da IN SGD/ME nº 94/2022.
    *   `{{definicao_necessidade}}`: Desenvolve `justificativaNecessidade`.
    *   `{{necessidades_negocio}}`: Desenvolve necessidades de negócio com base em `justificativaNecessidade` e `objetivoGeral`.
    *   `{{requisitos_contratacao}}`: Desenvolve requisitos técnicos e funcionais detalhados, baseados nos aceleradores e na `proposta_tecnica_content`.
    *   `{{levantamentomercado}}`: Analisa mercado, diferenciais da Xertica.ai usando `gcs_accelerator_content`, `proposta_tecnica_content`, e exemplos como PROCERGS MPRS TJES.
    *   `{{estimativa_demanda}}`: Estima demanda com base na `justificativaNecessidade` e `objetivoGeral`.
    *   `{{mapa_comparativo_custos}}`: Tabela e texto justificando a estimativa de custos.
    *   `{{estimativa_custo_total}}`: Valor final e justificativa.
    *   `{{descricao_solucao}}`: Descrição da solução, baseada nos aceleradores e `proposta_tecnica_content`.
    *   `{{justificativa_tecnica}}`: Justificativa para parcelamento (`parcelamentoContratacao`, `justificativaParcelamento`).
    *   `{{providencias_a_serem_tomadas}}`: Lista de providências (genéricas).
    *   `{{declaracao_viabilidade}}`: Declaração de viabilidade.
    *   `{{responsaveis_etp}}`: Nomes e cargos de responsáveis.
    *   `{{data_local_aprovacao}}`: Local e data para aprovação.
    *   `{{local_tr}}`: Cidade e UF para o TR.
    *   `{{data_tr}}`: Data para o TR.
    *   `{{processo_administrativo_tr_numero}}`: Número do processo administrativo (genérico).
    *   `{{cnae_tr_sugestao}}`: Sugestão de CNAE.
    *   `{{vigencia_tr_prazo}}`: Prazo de vigência.
    *   `{{subcontratacao_tr}}`: Regras de subcontratação.
    *   `{{garantia_tr}}`: Regras de garantia.
    *   `{{medição_pagamento_dias_recebimento}}`: Dias para recebimento.
    *   `{{medição_pagamento_dias_faturamento}}`: Dias para faturamento.
    *   `{{medição_pagamento_dias_pagamento}}`: Dias para pagamento.
    *   `{{critério_julgamento_tr}}`: Critério de julgamento.
    *   `{{metodologia_implementacao_tr}}`: Metodologia de implementação.
    *   `{{critérios_aceitacao_tr}}`: Critérios de aceitação.
    *   `{{obrigações_contratado_tr}}`: Obrigações do contratado.
    *   `{{obrigações_orgao_tr}}`: Obrigações do orgão.
    *   `{{gestao_contrato_tr}}`: Gestão do contrato.
    *   `{{sancoes_tr}}`: Sanções administrativas.
    *   `{{anexos_tr}}`: Anexos.
    *   `{{esfera_administrativa_tr}}`: Esfera administrativa.

    ---
    **MODELO DE ETP PARA PREENCHIMENTO (Siga esta estrutura rigorosamente):**
    ```markdown
    # Estudo Técnico Preliminar

    Contratação de solução tecnológica para {{acelerador_list_summary}}

    Processo Administrativo nº {{numero_processo_administrativo}}

    {{local_etp}}, {{mes_extenso}} de {{today.year}}

    ## Histórico de Revisões
    | Data | Versão | Descrição | Autor |
    |---|---|---|---|
    | {today.strftime('%d/%m/%Y')} | 1.0 | Finalização da primeira versão do documento | IA Xertica.ai |

    ## Área requisitante
    <Identificação da área requisitante e dos respectivos responsáveis>
    Identificação da Área requisitante: **{orgao_nome}**
    Nome do Responsável: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**
    Matrícula: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**

    ## Introdução
    O Estudo Técnico Preliminar – ETP é o documento constitutivo da primeira etapa do planejamento de uma contratação, que caracteriza o interesse público envolvido e a sua melhor solução. Ele serve de base ao Termo de Referência a ser elaborado, caso se conclua pela viabilidade da contratação.

    O ETP tem por objetivo identificar e analisar os cenários para o atendimento de demanda registrada no Documento de Formalização da Demanda – DFD, bem como demonstrar a viabilidade técnica e econômica das soluções identificadas, fornecendo as informações necessárias para subsidiar a tomada de decisão e o prosseguimento do respectivo processo de contratação.
    {contexto_geral_orgao if contexto_geral_orgao else f"A {esfera_administrativa} {orgao_nome}, em sua missão de modernizar a gestão pública e aprimorar a prestação de serviços ao cidadão, busca constantemente soluções inovadoras que garantam eficiência, transparência e segurança."}

    Referência: Inciso XI, do art. 2º e art. 11 da IN SGD/ME nº 94/2022.

    ## Descrição do problema e das necessidade
    A {esfera_administrativa} {orgao_nome} busca, por meio da iniciativa '{titulo_projeto}', endereçar um desafio crítico identificado: **{justificativa_necessidade}**.

    ## Necessidades do negócio
    A contratação visa suprir as seguintes necessidades do negócio do {orgao_nome}, com impactos diretos na eficiência operacional e na entrega de serviços:
    - Otimização de processos internos e externos.
    - Melhoria da experiência do usuário e do cidadão.
    - Aumento da produtividade das equipes.
    - Tomada de decisões mais assertivas com base em dados.
    - Garanta a transparência e compliance.

    ## Requisitos da Contratação
    Os requisitos gerais para a contratação da solução são:
    - **Requisitos Funcionais:** A solução deve {justificativa_necessidade} e auxiliar o {orgao_nome} a {objetivo_geral}.
    - **Requisitos Não Funcionais:** A solução deve garantir segurança dos dados, escalabilidade, alta disponibilidade e fácil integração com sistemas existentes do {orgao_nome}.
    - A solução deve atender às especificações detalhadas na proposta técnica da Xertica.ai.

    ## Levantamento de mercado
    {levantamentomercado}

    ## Estimativa de demanda - quantidade de bens e serviços
    A estimativa de demanda para os serviços/bens objeto desta contratação será:
    - Quantitativos: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**. Serão definidos em detalhe no Termo de Referência com base na análise do volume de {tipo_demanda_sugerida: interações, documentos, usuários} e na capacidade de processamento dos aceleradores da Xertica.ai.

    ## Mapa comparativo dos custos
    O mapa comparativo de custos detalhado está disponível na Proposta Comercial da Xertica.ai e em documentos de pesquisa de mercado. A análise considerou [parâmetros de comparação: TCO, ROI, custos de implementação, etc.]. O valor estimado foi ratificado por:
    {price_map_to_use_template}

    ## Estimativa de custo total da contratação
    O valor estimado global para esta contratação é de **R$ {valor_estimado if valor_estimado is not None else '[A SER DEFINIDO PELO ÓRGÃO/ADMINISTRAÇÃO]'}**.

    ## Descrição da solução como um todo
    A solução proposta abrange a implementação dos aceleradores de Inteligência Artificial da Xertica.ai, com foco em {', '.join(llm_context_data.get('produtosXertica', []))}, que em conjunto formam uma plataforma integrada para {objetivo_geral}. A proposta técnica da Xertica.ai (anexada) detalha a arquitetura, os componentes, e os serviços de implementação e suporte contínuo da solução.

    ## Justificativa do parcelamento ou não da contratação
    **Decisão sobre Parcelamento:** **{parcelamento_contratacao}**.
    **Justificativa:** {justificativa_parcelamento if parcelamento_contratacao == 'Justificar' and justificativa_parcelamento else f"A decisão por {('parcelar' if parcelamento_contratacao == 'Sim' else 'não parcelar')} a contratação visa {('otimizar a gestão de recursos e adaptar o projeto por fases, permitindo entregas incrementais.' if parcelamento_contratacao == 'Sim' else 'garantir a integralidade da solução e a sinergia entre seus componentes, otimizando o processo de implementação e a entrega de resultados completos.')}."}

    ## Providências a serem tomadas
    As providências a serem tomadas para a plena execução da contratação incluem:
    1.  Formalização do processo de contratação e assinatura do contrato.
    2.  Definição de cronograma detalhado de implantação.
    3.  Alocação de equipe técnica do {orgao_nome} para acompanhamento e validação.
    4.  Realização das etapas de capacitação e transferência de conhecimento.
    5.  Configuração e personalização da solução Xertica.ai.
    6.  Implantação em ambiente de produção e acompanhamento pós-implantação.

    ## Declaração de viabilidade
    Após a análise técnica e econômica preliminar, declara-se a viabilidade da contratação da solução Xertica.ai para atendimento da necessidade e objetivo do {orgao_nome}. A solução demonstra ser a mais adequada, tecnológica e economicamente vantajosa, alinhada às políticas e estratégias do {esfera_administrativa} {orgao_nome}.

    ## Responsáveis
    <Identificar os responsáveis pela elaboração do ETP>
    A Equipe de Planejamento da Contratação foi instituída pela Portaria nº **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]** (ou outro instrumento equivalente de formalização), de {today.day} de {mes_extenso} de {today.year}.
    <Conforme o § 2º do Art. 11 da IN SGD/ME nº 94, de 2022, o Estudo Técnico Preliminar deverá ser aprovado e assinado pelos Integrantes Técnicos e Requisitantes e pela autoridade máxima da área de TIC>

    INTEGRANTE TÉCNICO                   INTEGRANTE REQUISITANTE

    **[Nome do Integrante Técnico]**            **[Nome do Integrante Requisitante]**
    Matrícula/SIAPE: **[Matrícula/SIAPE]**              Matrícula/SIAPE: **[Matrícula/SIAPE]**

    ## Aprovação e declaração de conformidade
    <Aprovação do documento e declaração expressa da autoridade máxima da Área de TIC quanto à adequação dos estudos realizados.
    Aprovo este Estudo Técnico Preliminar e atesto sua conformidade às disposições da Instrução Normativa SGD/ME nº 94, de 23 de dezembro de 2022.

    {local_etp_full}, {today.day} de {mes_extenso} de {today.year}
    ```

    ---
    **MODELO DE TR PARA PREENCHIMENTO (Siga esta estrutura rigorosamente):**

    ```markdown
    # Termo de Referência – Nº [A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]

    **{modelo_licitacao_text_for_tr}**
    <Identificação do Processo Administrativo>

    ## 1 – DEFINIÇÃO DO OBJETO
    A presente contratação tem como objeto {justificativa_necessidade} por meio da implementação da solução Xertica.ai, com o objetivo de {objetivo_geral}, garantindo a eficiência e a modernização dos serviços do {orgao_nome}, conforme condições e exigências estabelecidas neste instrumento.

    1.1. Contratação de serviços estratégicos de Tecnologia da Informação baseados em Inteligência Artificial, caracterizados como [Bens ou Serviços Comuns/Especiais], para atender às necessidades permanentes e contínuas de modernização da {orgao_nome}.
    *   Ramo de Atividade predominante da contratação: {cnae_tr_sugestao}. O CNAE específico deverá ser confirmado e preenchido pelo órgão.
    *   Quantitativos estimados: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**. Serão definidos no contrato com base nas necessidades específicas e no dimensionamento da solução Xertica.ai.
    *   Prazo do contrato: O contrato terá vigência de **{prazos_estimados}**, contados a partir da assinatura, podendo ser prorrogado conforme a Lei nº 14.133/2021.

    ## 2 – FUNDAMENTAÇÃO DA CONTRATAÇÃO
    A Fundamentação da Contratação e de seus quantitativos encontra-se pormenorizada em tópico específico dos Estudos Técnicos Preliminares, que é anexo a este Termo de Referência.

    2.1. O objeto da contratação está previsto no Plano de Contratações Anual {today.year}, conforme detalhamento a seguir:
    *   ID PCA no PNCP e/ou SGA: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**
    *   Data de publicação no PNCP: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**
    *   Id do item no PCA: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**

    2.2. Justificativa da contratação: A contratação da solução Xertica.ai é fundamental para {justificativa_necessidade} no {orgao_nome}, buscando a otimização dos processos de trabalho e a melhoria contínua dos serviços prestados.

    2.3. Enquadramento da contratação: A contratação fundamenta-se no {legal_justification_llm_simulated} e nas demais normas legais e regulamentares atinentes à matéria, em plena observância à Lei nº 14.133/2021.

    ## 3 – DESCRIÇÃO DA SOLUÇÃO COMO UM TODO
    A solução a ser contratada consiste na implementação e suporte dos aceleradores de Inteligência Artificial da Xertica.ai, que se mostraram a alternativa mais vantajosa e completa para {objetivo_geral} no {orgao_nome}.

    3.1. O objeto da contratação compreende:
    *   A disponibilização e configuração dos aceleradores Xertica.ai selecionados ({', '.join(llm_context_data.get('produtosXertica', []))}).
    *   Serviços de implementação, customização e integração.
    *   Suporte técnico especializado e manutenção continuada.
    *   Treinamento e capacitação da equipe do {orgao_nome}.
    *   Garantia de atualização tecnológica e inovação.

    3.2. Forma de execução da contratação: indireta, em regime de **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - Ex: empreitada por preço global; preço unitário; tarefa; escopo]**.

    3.3. A descrição detalhada da solução como um todo encontra-se pormenorizada no Estudo Técnico Preliminar e na Proposta Técnica da Xertica.ai (anexo).

    ## 4 – REQUISITOS DA CONTRATAÇÃO
    Os requisitos necessários à contratação são essenciais para o atendimento da necessidade especificada e a garantia da qualidade e desempenho da solução.

    4.1. Os requisitos necessários para a presente contratação são:
    *   **Requisitos Funcionais:** A solução deve atender às funcionalidades específicas de cada acelerador conforme suas documentações (Battle Cards e Data Sheets) e a aplicação detalhada no ETP.
    *   **Requisitos Não Funcionais:**
        *   **Segurança:** Conformidade com a LGPD e padrões de segurança de dados.
        *   **Escalabilidade:** Capacidade de escalar para atender a demanda crescente.
        *   **Disponibilidade:** Mínimo de 99.5% de disponibilidade garantida por SLA.
        *   **Integração:** Capacidade de integração com sistemas legados do {orgao_nome}.
    *   **Práticas de Sustentabilidade:** Alinhamento com critérios de sustentabilidade ambiental, social e econômica.

    4.2. Da exigência de carta de solidariedade: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - (SIM/NÃO) e justificativa em edital]**.

    4.3. SUBCONTRATAÇÃO: **{subcontratacao_tr}**.

    4.4. GARANTIA DA CONTRATAÇÃO: **{garantia_tr}**.

    4.5. O Contratado deverá realizar a transição contratual com transferência de conhecimento, tecnologia e técnicas empregadas.

    4.6. VISTORIA: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - (SIM/NÃO) e detalhes]**.

    ## 5 – EXECUÇÃO DO OBJETO
    A execução do objeto será realizada de forma indireta, com foco na entrega de resultados e na operação contínua da solução.

    5.1. O prazo de prestação dos serviços e entrega das soluções será definido nos termos da Ordem de Serviço, emitidas conforme cronograma detalhado na Proposta Técnica da Xertica.ai.

    5.2. Os serviços serão executados e as soluções disponibilizadas remotamente, com possibilidade de visitas técnicas presenciais, se necessário.

    5.3. Deverão ser observados os métodos, rotinas e procedimentos de implementação e suporte definidos na Proposta Técnica da Xertica.ai.

    5.4. A CONTRATADA deverá disponibilizar todos os materiais, equipamentos e ferramentas necessárias para a perfeita execução dos serviços.

    5.5. O prazo de garantia contratual dos serviços será de, no mínimo, **12 (doze) meses**, contado a partir do recebimento definitivo do objeto.

    ## 6 – GESTÃO DO CONTRATO
    A gestão do contrato será realizada por um fiscal do {orgao_nome}, que será responsável por acompanhar a execução do objeto, verificar o cumprimento das obrigações e aprovar os pagamentos, conforme as normas aplicáveis.

    6.1. O contrato deverá ser executado fielmente pelas partes, de acordo com as cláusulas avençadas e as normas da Lei nº 14.133, de 2021.
    6.2. As comunicações entre o órgão e a contratada devem ser realizadas por escrito.
    6.3. O CONTRATANTE poderá convocar representante da empresa para adoção de providências imediatas.
    6.4. A formalização da contratação ocorrerá por meio de termo de contrato.
    6.5. Após a assinatura, haverá reunião inicial para apresentação do plano de fiscalização.
    6.6. A execução do contrato será acompanhada e fiscalizada pelo(s) fiscal(is) do contrato.
    6.7. A CONTRATADA deverá manter preposto para representá-la na execução do contrato.

    ## 7 – MEDIÇÃO E PAGAMENTO
    A medição e pagamento serão realizados com base na entrega e no atingimento dos Acordos de Níveis de Serviço (ANS) ou Instrumentos de Medição de Resultados (IMR) associados à solução.

    7.1. A avaliação da execução do objeto utilizará o Instrumento de Medição de Resultado (IMR), conforme prescrições: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - metodologia e regras]**.
    7.2. O valor devido a título de pagamento mensal à CONTRATADA será mensurado pelos indicadores do IMR.
    7.3. O recebimento dos serviços será provisório e definitivo, conforme Art. 140 da Lei nº 14.133/2021.
    7.4. Do Faturamento: A CONTRATADA deverá apresentar fatura ou nota fiscal (em {dias_faturamento_tr} dias úteis) após comunicação do gestor, com comprovações de regularidade fiscal e trabalhista.
    7.5. Das condições de pagamento: O pagamento será efetuado em {dias_pagamento_tr} dias corridos após o atesto da Fatura/Nota Fiscal.

    ## 8 – SELEÇÃO DO FORNECEDOR
    A seleção do fornecedor será realizada conforme a modalidade de contratação definida, garantindo a lisura e a adequação legal do processo.

    8.1. O fornecedor será selecionado por meio de **{modelo_licitacao_tr_text}**, com adoção do critério de julgamento pelo **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO - Ex: MENOR PREÇO, MELHOR TÉCNICA]**.
    *   **Exigências de Habilitação:** Serão observados os requisitos exigidos no Aviso de Contratação ou Edital, em conformidade com a Lei nº 14.133/2021.

    ## 9 – ESTIMATIVA DO PREÇO
    A estimativa de preço baseia-se na Proposta Comercial da Xertica.ai e na análise de mercado, sendo um valor de referência para a contratação.

    9.1. O valor estimado da contratação é de **R$ {valor_estimado if valor_estimado is not None else '[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]'}**. A memória de cálculo está disponível em documento separado.

    ## 10 – ADEQUAÇÃO ORÇAMENTÁRIA
    A adequação orçamentária para a presente contratação será assegurada pelos recursos consignados na Lei Orçamentária Anual do {orgao_nome} ({esfera_administrativa_tr} {orgao_nome.split(" ")} ).

    10.1. As despesas decorrentes da presente contratação correrão à conta de recursos específicos consignados no Orçamento Geral da União (ou Estadual/Municipal), mediante a seguinte dotação:
    *   UG Executora: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**
    *   Programa de Trabalho: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**
    *   Fonte: **[A SER PREENCHIDO PELO ÓRGÃO/ADMINISTRAÇÃO]**
    *   Natureza da Despesa: {cnae_tr_sugestao}

    10.2. A dotação relativa aos exercícios financeiros subsequentes será indicada após aprovação da Lei Orçamentária respectiva e liberação dos créditos correspondentes, mediante apostilamento.

    **Há anexos no pedido:** Sim (Proposta Comercial e Proposta Técnica da Xertica.ai)

    {local_etp_full}
    ```

    ---
    Gere o objeto JSON agora, seguindo todas as instruções e o formato especificado.
    """

    try:
        response = await gemini_model.generate_content_async(
            llm_prompt_content,
            generation_config=_generation_config
        )
        
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
    prazosEstimados: str = Form(..., description="Prazos estimados para implantação e execução. Ex: 3 meses para implantação, 12 meses de operação."),
    modeloLicitacao: str = Form(...),
    parcelamentoContratacao: str = Form(...),
    contextoGeralOrgao: Optional[str] = Form(None),
    valorEstimado: Optional[float] = Form(None),
    justificativaParcelamento: Optional[str] = Form(None),
    propostaComercialFile: Optional[UploadFile] = File(None, description="Proposta Comercial da Xertica.ai em PDF."),
    propostaTecnicaFile: Optional[UploadFile] = File(None, description="Proposta Técnica da Xertica.ai em PDF.")
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
    # ATUALIZADO: Nome do arquivo do MoU do SERPRO para incluir '(1)'
    llm_context_data['gcs_legal_context_content']["MTI_CONTRATO_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MTI/CONTRATO DE PARCERIA 03-2024-MTI - XERTICA - ASSINADO.txt")
    llm_context_data['gcs_legal_context_content']["MPAP_ATA_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/MPAP/ATA DE REGISTRO DE PREÇOS Nº 041-2024-XERTICA.txt")
    llm_context_data['gcs_legal_context_content']["RISK_ANALYSIS_CONTEXT.txt"] = get_gcs_file_content("Detecção e Análise de Riscos/Detecção de análise de riscos.txt")
    llm_context_data['gcs_legal_context_content']["SERPRO_MOU_EXEMPLO.txt"] = get_gcs_file_content("Formas ágeis de contratação/Serpro/[Xertica & Serpro] Memorando de Entendimento (MoU) - VersãoFinal (1).txt") 

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
        new_doc_metadata = drive_service.files().create(body=new_doc_body, fields='id').execute() 
        document_id = new_doc_metadata.get('id') 
        
        if not document_id:
            raise HTTPException(status_code=500, detail="Falha ao criar novo documento no Google Docs.")
        logger.info(f"Documento único criado: {document_id}")

        # Inserir o conteúdo completo do ETP e TR no documento, aplicando formatação básica.
        combined_markdown = f"{etp_content}\n\n---\n\n{tr_content}" 
        # AQUI É A GRANDE MUDANÇA: Usando o parser básico de Markdown
        requests_docs_api = apply_basic_markdown_to_docs_requests(combined_markdown)
        
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests_docs_api}).execute()
        logger.info(f"Conteúdo ETP/TR inserido no documento {document_id} com formatação básica.")

        # 6. Definir Permissão de Leitura Pública e Obter Link
        permission = {
            'type': 'anyone', 
            'role': 'writer' # ALTERADO: Permissão para "writer" (editável)
        }
        drive_service.permissions().create(fileId=document_id, body=permission, fields='id').execute()
        logger.info(f"Permissões de escrita pública definidas para {document_id}. CUIDADO: Documento editável por qualquer um com o link.")

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
        exception_to_log = f"Erro na API do Google Docs/Drive: {error_details}"
        logger.exception(exception_to_log) 
        raise HTTPException(status_code=e.resp.status, detail=f"Erro na API do Google Docs/Drive: {error_details}")
    except Exception as e:
        logger.exception(f"Erro na geração do documento.") 
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}. Verifique os logs para mais detalhes.")

# Para executar localmente para testes (descomente para usar):
# if __name__ == "__main__":
#     import uvicorn
#     # Configure as variáveis de ambiente em um arquivo .env na raiz do projeto
#     # Ex: GCP_PROJECT_ID="seu-projeto-gcp", GCP_PROJECT_LOCATION="us-central1", GCS_BUCKET_NAME="seu-bucket-gcs"
#     # Ou configure suas credenciais ADC localmente via 'gcloud auth application-default-login'
#     # (Instale google-auth-oauthlib para isso)
#     uvicorn.run(app, host="0.0.0.0", port=8000)
