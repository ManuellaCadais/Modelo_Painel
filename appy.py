import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import numpy as np

# ==========================================================
# 1. CONFIGURAÇÕES E CSS (THEME: VERMELHO/DOURADO/BRANCO)
# ==========================================================
st.set_page_config(page_title="Dashboard - Centro Esportivo Integrado", page_icon="🏟️", layout="wide")

# Paleta Personalizada
PRIMARY_COLOR = "#F4CF03"    # Dourado
SECONDARY_COLOR = "#C3281E"  # Vermelho
TEXT_COLOR = "#FFFFFF"       # Branco
BG_COLOR = "#000000"         # Preto

# Sequência de cores para os gráficos
CUSTOM_PALETTE = ['#C3281E', '#F4CF03', '#FFFFFF', '#808080', '#A90F0B']

st.markdown(
    f"""
    <style>
    /* FUNDO GERAL */
    .stApp {{
        background-color: {BG_COLOR};
    }}
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.9);
        border-right: 2px solid {SECONDARY_COLOR};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {PRIMARY_COLOR} !important;
        font-family: 'Arial', sans-serif;
    }}
    [data-testid="stSidebar"] label {{
        color: {TEXT_COLOR} !important;
        font-weight: bold;
    }}
    
    /* TAGS (Multiselect) */
    span[data-baseweb="tag"] {{
        background-color: {SECONDARY_COLOR} !important;
        border: 1px solid {PRIMARY_COLOR} !important;
        border-radius: 4px !important;
    }}
    span[data-baseweb="tag"] span {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* TEXTOS E METRICS */
    h1, h2, h3, h4, h5, p, span, div {{
        color: white;
    }}
    div[data-testid="stMetric"] {{
        background-color: rgba(20, 20, 20, 0.8);
        border: 1px solid {PRIMARY_COLOR};
        border-left: 5px solid {SECONDARY_COLOR};
        border-radius: 8px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {PRIMARY_COLOR} !important;
        font-weight: bold;
        font-size: 1.2rem !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_COLOR} !important;
        font-size: 2rem !important;
        font-weight: 700;
    }}

    /* BOTÕES */
    div.stButton > button {{
        background-color: {SECONDARY_COLOR} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {PRIMARY_COLOR} !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }}
    
    /* DIVISORES */
    hr {{
        border-color: {PRIMARY_COLOR} !important;
        opacity: 0.5;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def set_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        st.markdown(
            f"""<style>.stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.95)), url("data:image/png;base64,{bin_str}"); 
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}</style>""",
            unsafe_allow_html=True)
    except:
        pass

# Tenta carregar a imagem de fundo
set_bg('background.jpg') 

# ==========================================================
# 2. SEGURANÇA
# ==========================================================
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Alteração de Texto: Tela de Login
        st.markdown(f"<h2 style='color:{PRIMARY_COLOR}; text-align:center;'>🔐 Acesso Restrito</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size: 0.9em;'>Centro Esportivo Integrado</p>", unsafe_allow_html=True)
        
        pwd = st.text_input("Senha de Acesso:", type="password")
        if st.button("Entrar no Painel"):
            if pwd == "mengo" or pwd == "six2025": 
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Senha incorreta")
    st.stop()

# ==========================================================
# 3. MAPEAMENTOS
# ==========================================================
map_planos = {'Ativos': 'Ativos', 'Adimplentes': 'Adimplentes', 'VIP': 'VIP', 'PERSONAL': 'Personal', 'Suspensos': 'Suspensos', 'Evasão (churn)': 'Evasão (churn)'}
map_fat = {'Faturamento': 'FATURAMENTO', ' Receita Operacional': 'Receita Operacional', '   Mensalidade': 'Mensalidade', ' Receita Não-Operacional': 'Receita Não-Operacional', '   Publicidade': 'Publicidade', '   Sublocações': 'Sublocações'}
map_mp = {'MATÉRIA PRIMA': 'MATERIA PRIMA', '   Insumos': 'INSUMOS', '     Whey': 'Whey', '     Insumos Outros': 'Insumos Outros', '   Frete': 'FRETE'}
map_impostos = {'Impostos': 'IMPOSTOS', '   Faturamento': 'FATURAMENTO.1', '      IRPJ': 'IRPJ', '      ICMS': 'ICMS', '      CSLL': 'CSLL', '      PIS': 'PIS', '      COFINS': 'COFINS', '      ISS': 'ISS', '   Outros': 'Outros ', '   Folha': 'FOLHA', '      INSS': 'INSS', '      FGTS': 'FGTS', '      IRPF': 'IRPF'}
map_custos = {'Total Despesas': 'Total Despesas', 'Custos Operacionais': 'CUSTOS OPERACIONAIS', ' Folha': 'FOLHA.1', '    Folha Equipe': 'Folha Equipe', '    Pró-labore Sócio': 'Pro-labore Sócios', ' Indireitos Pessoais': 'INDIRETOS PESSOAL', '    Saúde Ocupacional': 'Saúde Ocupacional', '    Treinamento e Cursos': 'Treinamento e Cursos', '    Recrutamento': 'Recrutamento', '    Alimentação Equipe': 'Alimentação Equipe', '    Outros': 'Outros', ' Aluguel': 'ALUGUEL', '    Aluguel Loja': 'Aluguel Loja', '    Outros Imóveis Apoio': 'Outros Imóveis Apoio', ' Vale Transporte': 'VALE TRANSPORTE', ' 13º Salário': '13 SALARIO', ' Férias': 'FÉRIAS', ' Recisões': 'RESCISOES', ' Despesas Adiministrativas': 'DESPESAS ADMINISTRATIVAS', '    Telefone': 'Telefone', '    Internet': 'Internet', '    Material de escritório': 'Material de escritorio', '    Combustivel/Taxi': 'COMBUSTIVEL/ TAXI', '    Assinatura TV/SPOTIFY': 'ASSINATURA TV/ SPORTIFY/ NETFLIX', '    Viagens e estadia administrativa': 'Viagens e estadias administrativas', '    Seguros': 'Seguros', '    Contabilidade': 'Contabilidade', '    Jurídico': 'Juridico', '    Licença de softwares': 'Licenca de softwares', '    Suporte Informática': 'Suporte Informatica', '    Auditoria': 'Auditorias', '    Despesa com Eventos': 'Despesa com Eventos', '    Outros': 'Outros.1', ' Condomínio': 'CONDOMINIO', '    Condomínio': 'Condominio', '    Vigilância': 'Vigilancia', '    Jardinagem/Serviços Gerais': 'Jardinagem/Serviços Gerais/limpeza Vidros', '    Taxa Lixo': 'Taxa Lixo ', ' Serviços Terceirizados': 'SERVIÇOS TERCEIRIZADOS', '    Equipe (Professores PJ)': 'EQUIPE (PROFESSORES PJ)', '    DJ': 'DJ', '    Manobrista': 'MANOBRISTA', ' Despesas com limpeza': 'DESPESAS COM LIMPEZA/BELEZA', '    Material de Limpeza': 'Material de Limpeza', '    Material de Beleza': 'Material de Beleza', '    Dedetização': 'Dedetização', '    Lavanderia': 'Lavanderia', ' Despesas Indiretas': 'DESPESAS INDIRETAS', '    Água': 'Agua', '    Energia': 'Energia', ' Marketing': 'MARKETING', ' Reposição': 'REPOSIÇÃO', '    Fardamento': 'Fardamentos', '    Acessórios/Utensílios': 'ACESSÓRIOS/UTENSÍLIOS', '    Equipamentos/Mobiliário': 'Equipamentos/Mobiliário', ' Manutenção': 'MANUTENCAO', '    Manutenção': 'Manutenção', '    Frete': 'Frete Manutenção', ' Royalties': 'Royalties', ' IPTU': 'IPTU', ' Taxas': 'Tarifa Bancária', '    -Tarifa Bancária': 'Tarifa Bancária'}
map_lucro_op = {'LUCRO OPERACIONAL': 'LUCRO OPERACIONAL'}
map_lucratividade = {'LUCRATIVIDADE OPERACIONAL (%)': 'LUCRATIVIDADE OPERACIONAL (%)'}
map_invest = {'Retiradas, Reinvestimentos': 'Retiradas, REINVESTIMENTOS E EMPRÉSTIMOS', 'Retiradas de Sócios': 'RETIRADAS DE SOCIOS', 'Reinvestimentos': 'REINVESTIMENTOS'}
map_lucro_liq = {'Lucro Líquido (sem investimentos)': 'LUCRO LÍQUIDO (sem investimentos)', 'Lucro Líquido (descontando os investimentos)': 'LUCRO LÍQUIDO (descontando os investimentos)'}
map_saldo = {'Saldo em Caixa': 'Saldo em Caixa'}

# ==========================================================
# 4. GERAÇÃO DE DADOS FICTÍCIOS
# ==========================================================
@st.cache_data
def generate_fake_data():
    # Apenas Unidade Brasília
    unidades = ['Brasília'] 
    
    meses_ordem = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 
                   'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    
    data_rows = []
    
    # Gera a estrutura base
    for un in unidades:
        for mes in meses_ordem:
            row = {'Unidade': un, 'Mes_Nome': mes}
            data_rows.append(row)
            
    df = pd.DataFrame(data_rows)
    
    # Função auxiliar para gerar valores aleatórios
    def random_values(df, col_name, min_val, max_val, is_float=True):
        if is_float:
            df[col_name] = np.random.uniform(min_val, max_val, size=len(df))
        else:
            df[col_name] = np.random.randint(min_val, max_val, size=len(df))
            
    # Preencher colunas
    all_maps = [map_fat, map_mp, map_impostos, map_custos, map_lucro_op, 
                map_lucratividade, map_invest, map_lucro_liq, map_saldo, map_planos]
    
    cols_to_generate = set()
    for m in all_maps:
        for k, v in m.items():
            cols_to_generate.add(v)
            
    for col in cols_to_generate:
        if 'churn' in col.lower() or 'lucratividade' in col.lower() or '%' in col:
            random_values(df, col, 0.01, 0.15)
        elif 'ativos' in col.lower() or 'adimplentes' in col.lower() or 'vip' in col.lower():
            random_values(df, col, 100, 800, is_float=False)
        elif 'faturamento' in col.upper():
            random_values(df, col, 150000, 450000)
        elif 'lucro' in col.lower():
            random_values(df, col, 20000, 100000)
        else:
            random_values(df, col, 1000, 15000)

    if 'Ativos' in df.columns and 'Adimplentes' in df.columns:
        df['Ativos'] = df['Adimplentes'] + np.random.randint(10, 50, size=len(df))

    return df, df.copy()

df_base, df_planos = generate_fake_data()

# ==========================================================
# 5. SIDEBAR - FILTROS
# ==========================================================
with st.sidebar:
    try:
        st.image("logo.jpg", width=160) 
    except:
        # Alteração de Texto: Fallback caso não tenha logo
        st.markdown(f"<h2 style='text-align: center; color: {SECONDARY_COLOR}'>Centro Esportivo<br>Integrado</h2>", unsafe_allow_html=True)

    st.markdown("### 🔍 Filtros Gerenciais")
    
    # 1. UNIDADE
    opcoes_unidade = sorted(list(df_base['Unidade'].unique()))
    sel_unidade = st.multiselect("1. Unidade", opcoes_unidade, default=opcoes_unidade)
    
    # 2. MÊS
    lista_ordem = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    sel_mes = st.multiselect("2. Mês", lista_ordem, default=lista_ordem)

    st.markdown("---")
    st.markdown("### 📊 Indicadores Financeiros")

    sel_planos = st.multiselect("3. Planos & Alunos", list(map_planos.keys()))
    sel_fat = st.multiselect("4. Composição Faturamento", list(map_fat.keys()))
    sel_mp = st.multiselect("5. Matéria Prima", list(map_mp.keys()))
    sel_impostos = st.multiselect("6. Impostos", list(map_impostos.keys()))
    sel_custos = st.multiselect("7. Custos Operacionais", list(map_custos.keys()))
    sel_lucro_op = st.multiselect("8. Lucro Operacional", list(map_lucro_op.keys()))
    sel_lucratividade = st.multiselect("9. Margem (%)", list(map_lucratividade.keys()))
    sel_invest = st.multiselect("10. Investimentos", list(map_invest.keys()))
    sel_lucro_liq = st.multiselect("11. Resultado Líquido", list(map_lucro_liq.keys()))
    sel_saldo = st.multiselect("12. Caixa", list(map_saldo.keys()))

# ==========================================================
# 6. ENGINE DE GRÁFICOS
# ==========================================================
if not sel_unidade:
    st.warning("Selecione a Unidade Brasília.")
    st.stop()

df_base_f = df_base[df_base['Unidade'].isin(sel_unidade) & df_base['Mes_Nome'].isin(sel_mes)]
df_planos_f = df_planos[df_planos['Unidade'].isin(sel_unidade) & df_planos['Mes_Nome'].isin(sel_mes)]

# --- FORMATADORES ---
def format_currency_full(valor):
    if pd.isna(valor): return "R$ 0,00"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_currency_short(valor):
    if pd.isna(valor): return "R$ 0"
    if abs(valor) >= 1_000_000:
        return f"R$ {valor/1_000_000:.1f}M".replace(".", ",")
    elif abs(valor) >= 1_000:
        return f"R$ {valor/1_000:.0f}k"
    else:
        return f"R$ {int(valor)}"

def format_number_br(valor, is_percent=False):
    if is_percent:
        return f"{valor*100:.1f}%".replace(".", ",")
    else:
        return f"{int(valor)}"

def plot_chart(df, x_col, y_col, color_col, title, barmode='group', is_currency=True, is_percent=False):
    x_title = "Mês" if x_col == 'Mes_Nome' else "Unidade"
    
    if x_col == 'Mes_Nome':
        df[x_col] = pd.Categorical(df[x_col], categories=lista_ordem, ordered=True)
        df = df.sort_values(x_col)
        gap_size = 0.15
    else:
        gap_size = 0.2

    unique_x_count = df[x_col].nunique()
    initial_range = [-0.5, 5.5] if unique_x_count > 6 else None
    show_slider = unique_x_count > 6

    if is_currency:
        df['label_text'] = df[y_col].apply(format_currency_short)
        df['hover_text'] = df[y_col].apply(format_currency_full)
    elif is_percent:
        df['label_text'] = df[y_col].apply(lambda x: format_number_br(x, is_percent=True))
        df['hover_text'] = df['label_text']
    else:
        df['label_text'] = df[y_col].apply(lambda x: format_number_br(x, is_percent=False))
        df['hover_text'] = df['label_text']

    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode=barmode,
                 text='label_text', 
                 title=f"<b>{title.upper()}</b>", 
                 custom_data=['hover_text'],
                 color_discrete_sequence=CUSTOM_PALETTE)
    
    fig.update_traces(
        texttemplate='<b>%{text}</b>', 
        hovertemplate="<b>%{fullData.name}</b><br>Valor: %{customdata[0]}<extra></extra>",
        marker_line_width=0,
        textfont_size=14,
        textfont_color='white',
        textposition='outside',
        cliponaxis=False
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color='white', 
        height=550, 
        bargap=gap_size, 
        title_font_size=22,
        title_font_color=PRIMARY_COLOR,
        margin=dict(t=80, b=100, l=20, r=20), 
        xaxis_title=None, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(
        rangeslider=dict(visible=show_slider, thickness=0.05, bgcolor="#222222"),
        range=initial_range,
        fixedrange=False,
        tickfont_size=12,
        tickfont_weight="bold"
    )
    
    if show_slider:
        fig.add_annotation(
            text="↔ Arraste para zoom ↔",
            xref="paper", yref="paper", x=0.5, y=-0.25,
            showarrow=False, font=dict(size=12, color=PRIMARY_COLOR)
        )
    
    fig.update_yaxes(showticklabels=False, gridcolor='rgba(255,255,255,0.1)', fixedrange=True)
    st.plotly_chart(fig, use_container_width=True)

def render_metric_or_chart(label, value_total, df_source, col_name, is_currency=True, is_percent=False):
    multi_unidade = len(sel_unidade) > 1
    multi_mes = len(sel_mes) > 1
    
    if multi_unidade and multi_mes:
        df_chart = df_source.groupby(['Mes_Nome', 'Unidade'])[col_name].sum().reset_index()
        plot_chart(df_chart, 'Mes_Nome', col_name, 'Unidade', label, is_currency=is_currency, is_percent=is_percent)
    
    elif multi_mes:
        df_chart = df_source.groupby('Mes_Nome')[col_name].sum().reset_index()
        df_chart['Unidade'] = sel_unidade[0] 
        plot_chart(df_chart, 'Mes_Nome', col_name, 'Unidade', label, is_currency=is_currency, is_percent=is_percent)
    
    elif multi_unidade:
        val_agg = 'mean' if is_percent else 'sum'
        if val_agg == 'mean':
            df_chart = df_source.groupby('Unidade')[col_name].mean().reset_index()
        else:
            df_chart = df_source.groupby('Unidade')[col_name].sum().reset_index()
        plot_chart(df_chart, 'Unidade', col_name, 'Unidade', label, is_currency=is_currency, is_percent=is_percent)
    
    else:
        if is_currency: val_str = format_currency_full(value_total)
        elif is_percent: val_str = format_number_br(value_total, is_percent)
        else: val_str = format_number_br(value_total, False)
        st.metric(label, val_str)

def render_grid_section(title, selection_list, mapping_dict, df_source, is_currency=True, is_percent_override=False):
    if not selection_list: return
    st.subheader(f"🏅 {title}")
    
    for item in selection_list:
        col = mapping_dict.get(item)
        if col in df_source.columns:
            if is_percent_override:
                val_total = df_source[col].mean()
            else:
                val_total = df_source[col].sum()
            
            render_metric_or_chart(item, val_total, df_source, col, is_currency=is_currency, is_percent=is_percent_override)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<hr style='border-color:{PRIMARY_COLOR}'>", unsafe_allow_html=True)
            
    st.divider()

# ==========================================================
# 7. LAYOUT DO DASHBOARD
# ==========================================================
st.title("DASHBOARD EXECUTIVO - MODELO")
resumo_unidades = "Todas" if len(sel_unidade) == len(opcoes_unidade) else ', '.join(sel_unidade)
resumo_meses = "Ano Completo" if len(sel_mes) == len(lista_ordem) else ', '.join(sel_mes)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(f"📍 **Unidades:** {resumo_unidades}")
with col_info2:
    st.info(f"📅 **Período:** {resumo_meses}")

st.markdown("---")

# Seção PLANOS
if sel_planos:
    st.subheader("👥 Planos e Membros")
    for item in sel_planos:
        col = map_planos.get(item)
        is_perc = (item == 'Evasão (churn)')
        is_curr = False
        if col in df_planos_f.columns:
            val = df_planos_f[col].mean() if is_perc else df_planos_f[col].sum()
            render_metric_or_chart(item, val, df_planos_f, col, is_currency=is_curr, is_percent=is_perc)
            st.markdown("<hr>", unsafe_allow_html=True)
    st.divider()

# Renderização das Seções Financeiras
render_grid_section("Composição Faturamento", sel_fat, map_fat, df_base_f, is_currency=True)
render_grid_section("Matéria Prima", sel_mp, map_mp, df_base_f, is_currency=True)
render_grid_section("Tributação", sel_impostos, map_impostos, df_base_f, is_currency=True)
render_grid_section("Custos Operacionais", sel_custos, map_custos, df_base_f, is_currency=True)
render_grid_section("Resultado Operacional", sel_lucro_op, map_lucro_op, df_base_f, is_currency=True)
render_grid_section("Performance (%)", sel_lucratividade, map_lucratividade, df_base_f, is_currency=False, is_percent_override=True)
render_grid_section("Fluxo de Investimentos", sel_invest, map_invest, df_base_f, is_currency=True)
render_grid_section("Lucro Líquido", sel_lucro_liq, map_lucro_liq, df_base_f, is_currency=True)
render_grid_section("Posição de Caixa", sel_saldo, map_saldo, df_base_f, is_currency=True)

all_sels = (sel_planos or sel_fat or sel_mp or sel_impostos or sel_custos or 
            sel_lucro_op or sel_lucratividade or sel_invest or sel_lucro_liq or sel_saldo)

if not all_sels:
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.warning("👈 Selecione indicadores no menu lateral para construir sua análise.")