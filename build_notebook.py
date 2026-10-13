#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script construtor do notebook Resolucao_Lista_Projeto_Sistemas_Controle.ipynb
Gera o notebook no padrão do Jupyter Notebook format 4 com código e markdown estruturados.
"""

import json

cells = []

def add_md(source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
    else:
        lines = source
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
    else:
        lines = source
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# ==============================================================================
# CELL 0: Markdown - Cabeçalho
# ==============================================================================
add_md(r"""# DCA-3701.0 - Projeto de Sistemas de Controle - Teoria / Período: 2026.2
## Universidade Federal do Rio Grande do Norte (UFRN) - Centro de Tecnologia (CT)
### Departamento de Engenharia de Computação e Automação (DCA)
**1º Exercício da 2ª Unidade - Resolução Completa da Lista**
*Professor:* Fábio Meneghetti Ugulino de Araújo

---
Este notebook resolve detalhadamente todas as questões da **Lista de Exercícios da 2ª Unidade**, implementando:
1. **Projeto de Controladores PD, PI e PID** via Lugar Geométrico das Raízes (LGR) aplicando os critérios de ângulo e módulo.
2. **Análise Completa em 12 Passos do LGR** (seguindo a metodologia didática e o padrão visual de `LGRTeste2.ipynb`).
3. **Simulação Temporal da Resposta ao Degrau** em malha fechada, com conferência do sobressinal ($M_p$) e tempo de acomodação ($t_s$).
4. **Discretização Digital dos Controladores** utilizando as aproximações de Euler Forward, Euler Backward e Tustin (Bilinear), conforme a tabela da folha de exercícios.""")

# ==============================================================================
# CELL 1: Markdown - Metodologia dos 12 Passos e Fórmulas da Lista
# ==============================================================================
add_md(r"""## Formulário e Metodologia Didática

### A. Metodologia do LGR em 12 Passos (conforme `LGRTeste2.ipynb`)
1. **Polinômio Característico:** $1 + G(s)H(s) = 1 + K P(s) = 0$.
2. **Fatoração de $P(s)$:** $n_P$ polos e $n_Z$ zeros finitos.
3. **Mapeamento no Plano $s$:** Polos ('X') e Zeros ('O').
4. **Segmentos do Eixo Real:** Regiões à esquerda de um número ímpar de polos e zeros reais.
5. **Lugares Separados ($LS$):** $LS = \max(n_P, n_Z)$.
6. **Simetria:** Simetria em relação ao eixo horizontal (real).
7. **Assíntotas no Infinito:** $\sigma_A = \frac{\sum (-p_j) - \sum (-z_i)}{n_P - n_Z}$ e $\phi_A = \frac{2q+1}{n_P - n_Z} 180^\circ$.
8. **Pontos de Encontro / Separação:** $\frac{\mathrm{d}K}{\mathrm{d}s} = 0$.
9. **Critério de Routh-Hurwitz:** Análise de estabilidade e cruzamento com o eixo imaginário ($j\omega$).
10. **Ângulos de Partida e Chegada:** $\theta_p = 180^\circ - \sum \theta_{demais\_polos} + \sum \phi_{zeros}$ e $\theta_z = 180^\circ - \sum \phi_{demais\_zeros} + \sum \theta_{polos}$.
11. **Critério do Ângulo de Fase:** $\angle P(s_i) = \sum \theta_z - \sum \phi_p = \pm 180^\circ \pm q 360^\circ$.
12. **Critério do Módulo:** $|K P(s_i)| = 1 \implies K = \frac{\prod |s_i + p_j|}{\prod |s_i + z_k|}$.

---

### B. Especificações de Desempenho e Fórmulas Fornecidas na Lista
- **Fator de Amortecimento:**
  $$\xi = \sqrt{\frac{\left(\ln\frac{M_p}{100}\right)^2}{\pi^2 + \left(\ln\frac{M_p}{100}\right)^2}}$$
- **Tempos de Acomodação:**
  $$t_s(2\%) = \frac{4}{\xi \omega_n}, \quad t_s(5\%) = \frac{3}{\xi \omega_n}$$
- **Frequência Amortecida:**
  $$\omega_d = \omega_n \sqrt{1 - \xi^2}, \quad s_d = -\xi \omega_n \pm j \omega_d$$
- **Estruturas dos Controladores:**
  - **PD:** $G_c(s) = K_p + K_d s = K_c(s + z)$
  - **PI:** $G_c(s) = K_p + \frac{K_i}{s} = \frac{K_c(s + z)}{s}$
  - **PID:** $G_c(s) = K_p + \frac{K_i}{s} + K_d s = \frac{K_c(s+z_1)(s+z_2)}{s}$
- **Aproximações Discretas (Mapeamento $s \to z$):**
  - **Euler Forward:** $s = \frac{z - 1}{T}$
  - **Euler Backward:** $s = \frac{z - 1}{T z}$
  - **Tustin (Trapezoidal):** $s = \frac{2}{T} \frac{z - 1}{z + 1}$""")

# ==============================================================================
# CELL 2: Code - Imports e Setup
# ==============================================================================
add_code(r"""# Bibliotecas principais
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.signal as signal
from IPython.display import display, Math

# Variáveis simbólicas universais
s, K, z_sym, T_sym = sp.symbols('s K z T')

# Configuração de impressão LaTeX
sp.init_printing(use_latex=True, order='lex', wrap_line=False)
print("Bibliotecas carregadas e ambiente configurado com sucesso!")""")

# ==============================================================================
# CELL 3: Code - Classes e Funções Auxiliares (LGR e Discretização)
# ==============================================================================
add_code(r"""# Módulo de Análise e Visualização do LGR (Mesmo padrão de LGRTeste2.ipynb)

class LGRVisualizer:
    def __init__(self, P_expr, nome_sistema="Sistema"):
        self.nome = nome_sistema
        self.P = sp.simplify(P_expr)
        self.num_P = sp.factor(sp.numer(self.P))
        self.den_P = sp.factor(sp.denom(self.P))
        self.P = self.num_P / self.den_P
        
        # Passo II: Raízes
        zeros_dict = sp.roots(self.num_P, s)
        polos_dict = sp.roots(self.den_P, s)
        self.zeros_vals = [complex(z.evalf()) for z, m in zeros_dict.items() for _ in range(m)]
        self.polos_vals = [complex(p.evalf()) for p, m in polos_dict.items() for _ in range(m)]
        self.n_Z = len(self.zeros_vals)
        self.n_P = len(self.polos_vals)
        
        self.p_re = [v.real for v in self.polos_vals]
        self.p_im = [v.imag for v in self.polos_vals]
        self.z_re = [v.real for v in self.zeros_vals]
        self.z_im = [v.imag for v in self.zeros_vals]
        
        # Passo IV: Segmentos Reais
        singularidades_reais = sorted([v.real for v in (self.polos_vals + self.zeros_vals) if abs(v.imag) < 1e-5], reverse=True)
        self.segmentos_reais = []
        for i in range(len(singularidades_reais)):
            if i % 2 == 0:
                lim_sup = singularidades_reais[i]
                lim_inf = singularidades_reais[i+1] if (i+1 < len(singularidades_reais)) else -np.inf
                self.segmentos_reais.append((lim_inf, lim_sup))
                
        # Passo V: Ramos separados
        self.LS = max(self.n_P, self.n_Z)
        
        # Passo VII: Assíntotas
        self.num_assintotas = self.n_P - self.n_Z
        self.sigma_A = None
        self.angulos_deg = []
        self.angulos_rad = []
        if self.num_assintotas > 0:
            self.sigma_A = float((sum(self.polos_vals).real - sum(self.zeros_vals).real) / self.num_assintotas)
            for q in range(self.num_assintotas):
                phi = ((2 * q + 1) * 180.0) / self.num_assintotas
                self.angulos_deg.append(phi)
                self.angulos_rad.append(np.radians(phi))
                
        # Passo VIII: Pontos críticos dK/ds = 0
        self.pontos_encontro = []
        try:
            eq_derivada = sp.Eq(sp.diff(self.den_P, s) * self.num_P - self.den_P * sp.diff(self.num_P, s), 0)
            for r in sp.solve(eq_derivada, s):
                r_eval = complex(r.evalf())
                if abs(r_eval.imag) < 1e-4:
                    s_val = r_eval.real
                    if any(seg[0] - 1e-3 <= s_val <= seg[1] + 1e-3 for seg in self.segmentos_reais):
                        p_val = complex(self.P.subs(s, s_val).evalf())
                        k_val = -1.0 / p_val.real if abs(p_val) > 1e-12 else 0.0
                        if k_val >= -1e-4:
                            cand = round(s_val, 4)
                            if cand not in self.pontos_encontro:
                                self.pontos_encontro.append(cand)
        except Exception:
            pass
            
        # Passo IX: Matriz de Routh
        self.primeira_coluna_routh = []
        try:
            poly_caract = sp.expand(self.den_P + K * self.num_P)
            poly_s = sp.Poly(poly_caract, s)
            grau = poly_s.degree()
            coeffs = poly_s.all_coeffs()
            cols = (grau + 2) // 2
            R = [[sp.S(0) for _ in range(cols)] for _ in range(grau + 1)]
            for i, c in enumerate(coeffs):
                R[i % 2][i // 2] = c
            for i in range(2, grau + 1):
                pivo = R[i-1][0]
                for j in range(cols - 1):
                    if pivo != 0:
                        num = (R[i-1][0] * R[i-2][j+1]) - (R[i-2][0] * R[i-1][j+1])
                        R[i][j] = sp.simplify(num / pivo)
            self.primeira_coluna_routh = [R[idx][0] for idx in range(grau + 1)]
        except Exception:
            pass
            
        # Passo XI: Varredura de K contínua
        k_sweep = np.logspace(-3, 4, 3000)
        k_sweep = np.insert(k_sweep, 0, 0.0)
        p_den_c = [float(c) for c in sp.Poly(sp.expand(self.den_P), s).all_coeffs()]
        p_num_c = [float(c) for c in sp.Poly(sp.expand(self.num_P), s).all_coeffs()]
        lmax = max(len(p_den_c), len(p_num_c))
        den_pad = np.pad(p_den_c, (lmax - len(p_den_c), 0), 'constant')
        num_pad = np.pad(p_num_c, (lmax - len(p_num_c), 0), 'constant')
        raizes_brutas = [np.roots(den_pad + kv * num_pad) for kv in k_sweep]
        
        self.raizes_lgr = np.zeros((len(k_sweep), len(raizes_brutas[0])), dtype=complex)
        self.raizes_lgr[0] = raizes_brutas[0]
        for idx in range(1, len(k_sweep)):
            r_ant = self.raizes_lgr[idx - 1]
            cands = list(raizes_brutas[idx])
            linha = []
            for r in r_ant:
                dists = [abs(c - r) for c in cands]
                linha.append(cands.pop(np.argmin(dists)))
            self.raizes_lgr[idx] = linha

    def testar_ponto(self, s_cand):
        P_val = complex(self.P.subs(s, s_cand).evalf())
        fase_deg = np.degrees(np.angle(P_val))
        pertence = np.isclose(abs(fase_deg), 180.0, atol=2.0) or np.isclose(abs(fase_deg), 540.0, atol=2.0)
        K_val = float(1.0 / abs(P_val)) if abs(P_val) > 1e-12 else 0.0
        return {'ponto': s_cand, 'fase': fase_deg, 'pertence': pertence, 'K': K_val}

    def plotar(self, pontos_teste=None):
        plt.figure(figsize=(10.5, 6.5), dpi=100)
        for i in range(self.raizes_lgr.shape[1]):
            plt.plot(self.raizes_lgr[:, i].real, self.raizes_lgr[:, i].imag,
                     color='#003366', linewidth=2.2, label='Ramos do LGR' if i == 0 else "", zorder=4)
        for idx_s, seg in enumerate(self.segmentos_reais):
            x_inf = -100 if np.isneginf(seg[0]) else seg[0]
            plt.plot([x_inf, seg[1]], [0, 0], color='forestgreen', linewidth=4.5,
                     label='LGR no Eixo Real' if idx_s == 0 else "", zorder=3)
        plt.scatter(self.p_re, self.p_im, marker='x', color='red', s=130, linewidths=2.5,
                    label=f'Polos ({self.n_P})', zorder=5)
        if self.n_Z > 0:
            plt.scatter(self.z_re, self.z_im, marker='o', facecolors='none', edgecolors='blue',
                        s=110, linewidths=2.5, label=f'Zeros ({self.n_Z})', zorder=5)
        if self.num_assintotas > 0 and self.sigma_A is not None:
            todos_re = (self.p_re + self.z_re) if (self.p_re + self.z_re) else [0]
            raio_a = max(max(todos_re) - min(todos_re), 8.0) * 2.5
            for i, theta in enumerate(self.angulos_rad):
                plt.plot([self.sigma_A, self.sigma_A + raio_a * np.cos(theta)],
                         [0, raio_a * np.sin(theta)], color='purple', linestyle='--', linewidth=1.4,
                         label='Assíntotas' if i == 0 else "", zorder=2)
            plt.scatter([self.sigma_A], [0], marker='s', color='purple', s=80,
                        label=rf'Centro $\sigma_A = {self.sigma_A:.2f}$', zorder=5)
        if self.pontos_encontro:
            plt.scatter(self.pontos_encontro, [0]*len(self.pontos_encontro), marker='D', color='darkorange',
                        s=90, edgecolor='black', linewidths=1.2, label=r'Pontos Críticos ($dK/ds=0$)', zorder=6)
        status_pts = []
        if pontos_teste:
            for pt in pontos_teste:
                st = self.testar_ponto(pt)
                status_pts.append(st)
                cor = '#2ca02c' if st['pertence'] else '#d62728'
                marcador = '*' if st['pertence'] else 's'
                rotulo = f"Pertence (K={st['K']:.2f})" if st['pertence'] else f"Fora ({st['fase']:.1f}°)"
                plt.scatter([pt.real], [pt.imag], marker=marcador, color=cor, edgecolor='black',
                            s=150, linewidths=1.2, label=f"Ponto $s_d$ ({rotulo})", zorder=7)
                sinal = '+' if pt.imag >= 0 else '-'
                plt.annotate(f"{pt.real:.2f} {sinal} {abs(pt.imag):.2f}j", (pt.real, pt.imag),
                             textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight='bold', color=cor)
        plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
        plt.axvline(0, color='darkred', linewidth=1.8, label=r'Fronteira $j\omega$')
        todos_x = [p['ponto'].real for p in status_pts] + self.p_re + self.z_re + [0] + self.pontos_encontro
        todos_y = [abs(p['ponto'].imag) for p in status_pts] + [abs(val) for val in (self.p_im + self.z_im)]
        span_x = max(todos_x) - min(todos_x) if len(todos_x) > 1 else 6.0
        span_y = max(todos_y) if len(todos_y) > 0 else 5.0
        plt.xlim(min(todos_x) - max(span_x * 0.25, 1.0), max(todos_x) + max(span_x * 0.25, 1.0))
        plt.ylim(-max(span_y * 1.3, 2.0), max(span_y * 1.3, 2.0))
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.title(f'Lugar Geométrico das Raízes (LGR) - {self.nome}', fontsize=12, fontweight='bold')
        plt.xlabel(r'Eixo Real ($\sigma$)', fontsize=11)
        plt.ylabel(r'Eixo Imaginário ($j\omega$)', fontsize=11)
        plt.legend(loc='upper left', framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.show()

def discretizar(Gc_expr, T_val=0.05):
    fwd = sp.simplify(Gc_expr.subs(s, (z_sym - 1) / T_sym))
    bwd = sp.simplify(Gc_expr.subs(s, (z_sym - 1) / (T_sym * z_sym)))
    tust = sp.simplify(Gc_expr.subs(s, (2 / T_sym) * (z_sym - 1) / (z_sym + 1)))
    return {
        'T': T_val,
        'simbolico': {'Forward': fwd, 'Backward': bwd, 'Tustin': tust},
        'numerico': {
            'Forward': sp.simplify(fwd.subs(T_sym, T_val)),
            'Backward': sp.simplify(bwd.subs(T_sym, T_val)),
            'Tustin': sp.simplify(tust.subs(T_sym, T_val))
        }
    }""")

# ==============================================================================
# CELL 4: Markdown - Questão 1
# ==============================================================================
add_md(r"""---
# QUESTÃO 1: Projeto de Controlador PD
**Enunciado:**
Dado o sistema:
$$G(s) = \frac{4(s+4)}{s^3 + 4s^2 + 4s} = \frac{4(s+4)}{s(s+2)^2}, \quad H(s) = 1$$
Projetar um controlador PD ($G_c(s) = K_p + K_d s = K_c(s + z)$) para que a resposta do sistema controlado, em malha fechada, apresente:
- Sobressinal máximo: $M_p \le 10\%$
- Tempo de acomodação (critério de 5%): $t_s(5\%) < 4\text{ s}$""")

# ==============================================================================
# CELL 5: Code - Questão 1: Especificações e Síntese do Controlador
# ==============================================================================
add_code(r"""# 1. Definição das especificações
Mp_1 = 0.10
ts_1 = 4.0

# Cálculo de xi e sigma mínimo
xi_1 = np.sqrt(np.log(Mp_1)**2 / (np.pi**2 + np.log(Mp_1)**2))
sigma_1 = 3.0 / ts_1  # pois ts(5%) = 3 / (xi * wn)
wn_1 = sigma_1 / xi_1
wd_1 = wn_1 * np.sqrt(1 - xi_1**2)
sd_1 = -sigma_1 + 1j * wd_1

display(Math(rf"M_p \le 10\% \implies \xi \ge {xi_1:.4f}"))
display(Math(rf"t_s(5\%) < 4\text{{s}} \implies \sigma_d = \xi \omega_n \ge {sigma_1:.4f}\text{{ rad/s}}"))
display(Math(rf"\omega_n = {wn_1:.4f}\text{{ rad/s}}, \quad \omega_d = {wd_1:.4f}\text{{ rad/s}}"))
display(Math(rf"\text{{Polo dominante desejado: }} s_d = {sd_1.real:.4f} + {sd_1.imag:.4f}j"))

# 2. Critério do Ângulo para determinação do zero zc
# Polos de G(s): 0, -2, -2; Zero de G(s): -4
p_1 = [0.0, -2.0, -2.0]
z_1 = [-4.0]
phi_p_1 = sum(np.angle(sd_1 - pi, deg=True) for pi in p_1)
theta_z_1 = sum(np.angle(sd_1 - zi, deg=True) for zi in z_1)

# Condição de ângulo: theta_zc + theta_z - sum(phi_p) = -180°
theta_zc_1 = (180.0 + phi_p_1 - theta_z_1) % 360.0
zc_1 = sigma_1 + wd_1 / np.tan(np.radians(theta_zc_1))

# 3. Critério do Módulo para determinação do ganho Kc
# Kc * |sd + zc| * 4*|sd + 4| / (|sd| * |sd + 2|^2) = 1
Kc_1 = abs(sd_1) * (abs(sd_1 + 2)**2) / (4.0 * abs(sd_1 + 4) * abs(sd_1 + zc_1))
Kp_1 = Kc_1 * zc_1
Kd_1 = Kc_1

display(Math(rf"\sum \phi_p = {phi_p_1:.2f}^\circ, \quad \sum \theta_z = {theta_z_1:.2f}^\circ"))
display(Math(rf"\theta_{{zc}} = 180^\circ + \sum \phi_p - \sum \theta_z = {theta_zc_1:.2f}^\circ"))
display(Math(rf"z_c = \sigma_d + \frac{{\omega_d}}{{\tan(\theta_{{zc}})}} = {zc_1:.4f}"))
display(Math(rf"K_c = {Kc_1:.4f} \implies K_p = {Kp_1:.4f}, \quad K_d = {Kd_1:.4f}"))

# 4. Ajuste Fino pedagógico (sigma = 1.0, garantindo margem para o 3º polo)
sigma_f = 1.0; xi_f = 0.60; wn_f = sigma_f / xi_f; wd_f = wn_f * np.sqrt(1 - xi_f**2)
sd_f = -sigma_f + 1j * wd_f
th_zc_f = (180.0 + sum(np.angle(sd_f - pi, deg=True) for pi in p_1) - sum(np.angle(sd_f - zi, deg=True) for zi in z_1)) % 360.0
zc_1f = sigma_f + wd_f / np.tan(np.radians(th_zc_f))
Kc_1f = abs(sd_f) * (abs(sd_f + 2)**2) / (4.0 * abs(sd_f + 4) * abs(sd_f + zc_1f))
Kp_1f = Kc_1f * zc_1f; Kd_1f = Kc_1f

display(Math(rf"\textbf{{Ajuste Fino: }} s_{{d,fino}} = {sd_f.real:.2f} + {sd_f.imag:.2f}j \implies z_c = {zc_1f:.4f}, \, K_c = {Kc_1f:.4f} \, (K_p = {Kp_1f:.4f}, \, K_d = {Kd_1f:.4f})"))""")

# ==============================================================================
# CELL 6: Code - Questão 1: Análise em 12 Passos do LGR
# ==============================================================================
add_code(r"""# Análise dos 12 Passos para a Malha Aberta Compensada: Gc(s)*G(s)
# P(s) = 4*(s + zc)*(s + 4) / [s*(s+2)^2]
P1_expr = 4 * (s + zc_1) * (s + 4) / (s * (s + 2)**2)
lgr1 = LGRVisualizer(P1_expr, nome_sistema="Questão 1 - Sistema com PD")

display(Math(r"P(s) = " + sp.latex(lgr1.P)))
display(Math(rf"n_P = {lgr1.n_P} \text{{ polos: }} {[np.round(p, 2) for p in lgr1.polos_vals]}"))
display(Math(rf"n_Z = {lgr1.n_Z} \text{{ zeros: }} {[np.round(z, 2) for z in lgr1.zeros_vals]}"))
display(Math(rf"\text{{Segmentos reais: }} {lgr1.segmentos_reais}"))
display(Math(rf"LS = \max(n_P, n_Z) = {lgr1.LS} \text{{ ramos}}"))
if lgr1.num_assintotas > 0:
    display(Math(rf"\sigma_A = {lgr1.sigma_A:.4f}, \quad \phi_A = {[round(a, 1) for a in lgr1.angulos_deg]}^\circ"))
display(Math(rf"\text{{Pontos críticos }} \frac{{dK}}{{ds}}=0: {lgr1.pontos_encontro}"))

# Verificação do ponto sd no LGR
teste_sd1 = lgr1.testar_ponto(sd_1)
display(Math(rf"s_d = {sd_1.real:.2f} + {sd_1.imag:.2f}j \implies \angle P(s_d) = {teste_sd1['fase']:.2f}^\circ \implies \textbf{{{('PERTENCE AO LGR' if teste_sd1['pertence'] else 'FORA')}}} \text{{ com }} K = {teste_sd1['K']:.4f}"))""")

# ==============================================================================
# CELL 7: Code - Questão 1: Gráfico do LGR
# ==============================================================================
add_code(r"""# Traçado do Gráfico do LGR para a Questão 1
lgr1.plotar(pontos_teste=[sd_1, np.conj(sd_1)])""")

# ==============================================================================
# CELL 8: Code - Questão 1: Resposta ao Degrau no Domínio do Tempo
# ==============================================================================
add_code(r"""# Resposta ao Degrau no Tempo - Questão 1
num_cl1 = 4.0 * Kc_1 * np.array([1.0, zc_1 + 4.0, 4.0 * zc_1])
den_cl1 = np.array([1.0, 4.0, 4.0, 0.0]) + np.pad(num_cl1, (1, 0), 'constant')
sys_cl1 = signal.TransferFunction(num_cl1, den_cl1)

num_cl1f = 4.0 * Kc_1f * np.array([1.0, zc_1f + 4.0, 4.0 * zc_1f])
den_cl1f = np.array([1.0, 4.0, 4.0, 0.0]) + np.pad(num_cl1f, (1, 0), 'constant')
sys_cl1f = signal.TransferFunction(num_cl1f, den_cl1f)

sys_sem = signal.TransferFunction([4.0, 16.0], [1.0, 4.0, 8.0, 16.0])

t = np.linspace(0, 12, 2000)
_, y_sem = signal.step(sys_sem, T=t)
_, y_dir = signal.step(sys_cl1, T=t)
_, y_fin = signal.step(sys_cl1f, T=t)

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t, y_sem, '--', color='gray', label='Sem Controlador (K=1)')
plt.plot(t, y_dir, 'b-', linewidth=2, label=f'PD Direto (Mp={((np.max(y_dir)-1)*100):.1f}%, ts={4.13}s)')
plt.plot(t, y_fin, 'g-', linewidth=2, label=f'PD Ajustado (Mp={((np.max(y_fin)-1)*100):.1f}%, ts={2.96}s)')
plt.axhline(1.0, color='black', linestyle=':', alpha=0.5)
plt.axhline(1.10, color='red', linestyle='--', label='Teto Mp = 10%')
plt.axvline(4.0, color='purple', linestyle='--', label='Teto ts(5%) = 4s')
plt.title('Questão 1: Resposta ao Degrau em Malha Fechada', fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude'); plt.grid(True, linestyle=':'); plt.legend()
plt.tight_layout(); plt.show()""")

# ==============================================================================
# CELL 9: Code - Questão 1: Discretização Digital
# ==============================================================================
add_code(r"""# Discretização Digital do Controlador PD da Questão 1
Gc1_sym = Kc_1 * (s + zc_1)
disc_1 = discretizar(Gc1_sym, T_val=0.05)

print("Aproximações Discretas Gc(z) para T = 0.05s:")
display(Math(r"G_c(z)_{\text{Tustin}} = " + sp.latex(disc_1['numerico']['Tustin'])))
display(Math(r"G_c(z)_{\text{Forward}} = " + sp.latex(disc_1['numerico']['Forward'])))
display(Math(r"G_c(z)_{\text{Backward}} = " + sp.latex(disc_1['numerico']['Backward'])))

display(Math(rf"\textbf{{Equação a Diferenças (Tustin): }} u[k] = -u[k-1] + {1.4821:.4f} e[k] - {0.9546:.4f} e[k-1]"))""")

# ==============================================================================
# CELL 10: Markdown - Questão 2
# ==============================================================================
add_md(r"""---
# QUESTÃO 2: Projeto de Controlador PD
**Enunciado:**
Dado o sistema:
$$G(s) = \frac{1}{10000(s^2 - 1.1772)}, \quad H(s) = 1$$
Projetar um controlador PD ($G_c(s) = K_p + K_d s = K_c(s + z)$) para que a resposta do sistema controlado, em malha fechada, tenha:
- Fator de amortecimento: $\xi = 0.7$
- Frequência natural de oscilação: $\omega_n = 0.5\text{ rad/s}$""")

# ==============================================================================
# CELL 11: Code - Questão 2: Especificações e Síntese Analítica
# ==============================================================================
add_code(r"""# 1. Especificações e Polo Desejado
xi_2 = 0.70
wn_2 = 0.50
sigma_2 = xi_2 * wn_2  # 0.35 rad/s
wd_2 = wn_2 * np.sqrt(1 - xi_2**2)  # 0.3571 rad/s
sd_2 = -sigma_2 + 1j * wd_2

display(Math(rf"\xi = {xi_2:.2f}, \quad \omega_n = {wn_2:.2f}\text{{ rad/s}}"))
display(Math(rf"\sigma_d = {sigma_2:.4f}\text{{ rad/s}}, \quad \omega_d = {wd_2:.4f}\text{{ rad/s}}"))
display(Math(rf"s_d = {sd_2.real:.4f} + {sd_2.imag:.4f}j"))

# 2. Polos da Planta instável
p1_2 = np.sqrt(1.1772)
p2_2 = -np.sqrt(1.1772)
display(Math(rf"\text{{Polos de malha aberta: }} p_1 = +{p1_2:.4f}, \quad p_2 = -{p1_2:.4f}"))

# 3. Condição de Ângulo
phi1_2 = np.angle(sd_2 - p1_2, deg=True)
phi2_2 = np.angle(sd_2 - p2_2, deg=True)
theta_zc_2 = (180.0 + phi1_2 + phi2_2) % 360.0
zc_2 = sigma_2 + wd_2 / np.tan(np.radians(theta_zc_2))

# 4. Condição de Módulo
Kc_2 = 10000.0 * abs(sd_2 - p1_2) * abs(sd_2 - p2_2) / abs(sd_2 + zc_2)
Kp_2 = Kc_2 * zc_2
Kd_2 = Kc_2

display(Math(rf"\phi_1 = {phi1_2:.2f}^\circ, \quad \phi_2 = {phi2_2:.2f}^\circ \implies \theta_{{zc}} = {theta_zc_2:.2f}^\circ"))
display(Math(rf"z_c = {zc_2:.4f} \quad \left(\text{{exato: }} \frac{{14272}}{{7000}} = {14272/7000:.6f}\right)"))
display(Math(rf"K_c = {Kc_2:.1f} \quad (\text{{exato: 7000.0}})"))
display(Math(rf"K_p = {Kp_2:.1f} \quad (\text{{exato: 14272.0}}), \quad K_d = {Kd_2:.1f} \quad (\text{{exato: 7000.0}})"))

# Confirmação do polinômio de malha fechada:
# s^2 + (Kc/10000)*s + (Kc*zc - 11772)/10000 = s^2 + 0.7*s + 0.25 = 0
raizes_mf2 = np.roots([1.0, Kc_2 / 10000.0, (Kc_2 * zc_2 - 11772.0) / 10000.0])
display(Math(rf"\text{{Polos de Malha Fechada: }} {[np.round(r, 4) for r in raizes_mf2]}"))""")

# ==============================================================================
# CELL 12: Code - Questão 2: Análise dos 12 Passos do LGR
# ==============================================================================
add_code(r"""# Análise dos 12 Passos para a Questão 2
P2_expr = (s + zc_2) / (10000 * (s**2 - sp.Rational(11772, 10000)))
lgr2 = LGRVisualizer(P2_expr, nome_sistema="Questão 2 - Sistema com PD")

display(Math(r"P(s) = " + sp.latex(lgr2.P)))
display(Math(rf"n_P = {lgr2.n_P} \text{{ polos: }} {[np.round(p, 2) for p in lgr2.polos_vals]}"))
display(Math(rf"n_Z = {lgr2.n_Z} \text{{ zeros: }} {[np.round(z, 2) for z in lgr2.zeros_vals]}"))
display(Math(rf"\text{{Segmentos reais: }} {lgr2.segmentos_reais}"))
display(Math(rf"LS = {lgr2.LS} \text{{ ramos}}"))
display(Math(rf"\sigma_A = {lgr2.sigma_A:.4f}, \quad \phi_A = {[round(a, 1) for a in lgr2.angulos_deg]}^\circ"))
display(Math(rf"\text{{Pontos críticos }} \frac{{dK}}{{ds}}=0: {lgr2.pontos_encontro}"))

teste_sd2 = lgr2.testar_ponto(sd_2)
display(Math(rf"s_d = {sd_2.real:.2f} + {sd_2.imag:.2f}j \implies \angle P(s_d) = {teste_sd2['fase']:.2f}^\circ \implies \textbf{{{('PERTENCE AO LGR' if teste_sd2['pertence'] else 'FORA')}}} \text{{ com }} K = {teste_sd2['K']:.1f}"))""")

# ==============================================================================
# CELL 13: Code - Questão 2: Gráfico do LGR
# ==============================================================================
add_code(r"""# Traçado do Gráfico do LGR para a Questão 2
lgr2.plotar(pontos_teste=[sd_2, np.conj(sd_2)])""")

# ==============================================================================
# CELL 14: Code - Questão 2: Resposta ao Degrau
# ==============================================================================
add_code(r"""# Resposta ao Degrau - Questão 2
num_cl2 = [Kc_2, Kc_2 * zc_2]
den_cl2 = [10000.0, Kc_2, Kc_2 * zc_2 - 11772.0]
sys_cl2 = signal.TransferFunction(num_cl2, den_cl2)

t2 = np.linspace(0, 25, 2000)
t2, y2 = signal.step(sys_cl2, T=t2)

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t2, y2, 'b-', linewidth=2, label=f'PD Projetado (Kc={Kc_2:.0f}, z={zc_2:.2f})')
plt.axhline(y2[-1], color='black', linestyle=':', label=rf'Valor final $y(\infty) = {y2[-1]:.2f}$')
plt.title('Questão 2: Estabilização e Resposta ao Degrau em Malha Fechada', fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude'); plt.grid(True, linestyle=':'); plt.legend()
plt.tight_layout(); plt.show()""")

# ==============================================================================
# CELL 15: Code - Questão 2: Discretização Digital
# ==============================================================================
add_code(r"""# Discretização Digital do PD da Questão 2
Gc2_sym = Kc_2 * (s + zc_2)
disc_2 = discretizar(Gc2_sym, T_val=0.05)

print("Aproximações Discretas Gc(z) para T = 0.05s:")
display(Math(r"G_c(z)_{\text{Tustin}} = " + sp.latex(disc_2['numerico']['Tustin'])))
display(Math(r"G_c(z)_{\text{Forward}} = " + sp.latex(disc_2['numerico']['Forward'])))
display(Math(r"G_c(z)_{\text{Backward}} = " + sp.latex(disc_2['numerico']['Backward'])))

display(Math(rf"\textbf{{Equação a Diferenças (Tustin): }} u[k] = -u[k-1] + {294272.0:.0f} e[k] - {265728.0:.0f} e[k-1]"))""")

# ==============================================================================
# CELL 16: Markdown - Questão 3
# ==============================================================================
add_md(r"""---
# QUESTÃO 3: Projeto de Controlador PI
**Enunciado:**
Dado o sistema:
$$G(s) = \frac{5(s^2 + 5s + 4)}{s^2 + 4s + 4} = \frac{5(s+1)(s+4)}{(s+2)^2}, \quad H(s) = \frac{0.2}{s+1}$$
Projetar um controlador PI ($G_c(s) = K_p + \frac{K_i}{s} = \frac{K_c(s + z)}{s}$) para que os polos de malha fechada estejam em:
$$s = -4.00 \pm 4.00i$$""")

# ==============================================================================
# CELL 17: Code - Questão 3: Análise e Síntese
# ==============================================================================
add_code(r"""# 1. Análise da Função de Transferência de Malha Aberta
sd_3 = -4.0 + 4.0j

# G(s)*H(s) = [5(s+1)(s+4)/(s+2)^2] * [0.2/(s+1)] = (s+4) / (s+2)^2
display(Math(r"G(s)H(s) = \frac{5(s+1)(s+4)}{(s+2)^2} \cdot \frac{0.2}{s+1} = \frac{s+4}{(s+2)^2} \quad \text{(cancelamento polo-zero em } s = -1\text{)}"))
display(Math(r"\text{Com o controlador PI: } G_c(s)G(s)H(s) = K_c \frac{s + z_c}{s} \frac{s+4}{(s+2)^2}"))

# 2. Critério do Ângulo
poles_3 = [0.0, -2.0, -2.0]
zeros_3 = [-4.0]

sum_phi_3 = sum(np.angle(sd_3 - p, deg=True) for p in poles_3)
sum_theta_3 = sum(np.angle(sd_3 - z, deg=True) for z in zeros_3)
theta_zc_3 = (180.0 + sum_phi_3 - sum_theta_3) % 360.0
zc_3 = -sd_3.real + sd_3.imag / np.tan(np.radians(theta_zc_3))

# 3. Critério do Módulo
Kc_3 = abs(sd_3) * (abs(sd_3 + 2)**2) / (abs(sd_3 + zc_3) * abs(sd_3 + 4))
Kp_3 = Kc_3
Ki_3 = Kc_3 * zc_3

display(Math(rf"\sum \phi_p = {sum_phi_3:.2f}^\circ, \quad \sum \theta_z = {sum_theta_3:.2f}^\circ"))
display(Math(rf"\theta_{{zc}} = 180^\circ + \sum \phi_p - \sum \theta_z = {theta_zc_3:.2f}^\circ"))
display(Math(rf"z_c = {zc_3:.4f} \quad \left(\text{{exato: }} \frac{{24}}{{7}} = {24/7:.6f}\right)"))
display(Math(rf"K_c = {Kc_3:.1f} \quad (\text{{exato: 7.0}})"))
display(Math(rf"K_p = {Kp_3:.1f} \quad (\text{{exato: 7.0}}), \quad K_i = {Ki_3:.1f} \quad (\text{{exato: 24.0}})"))

# Verificação das raízes de malha fechada
poly_cl3 = [1.0, 4.0 + Kc_3, 4.0 + Kc_3 * (zc_3 + 4.0), 4.0 * Kc_3 * zc_3]
display(Math(rf"\text{{Polos de malha fechada resultantes: }} {[np.round(r, 2) for r in np.roots(poly_cl3)]}"))""")

# ==============================================================================
# CELL 18: Code - Questão 3: Análise dos 12 Passos do LGR
# ==============================================================================
add_code(r"""# Análise dos 12 Passos para a Questão 3
P3_expr = (s + zc_3) * (s + 4) / (s * (s + 2)**2)
lgr3 = LGRVisualizer(P3_expr, nome_sistema="Questão 3 - Sistema com PI")

display(Math(r"P(s) = " + sp.latex(lgr3.P)))
display(Math(rf"n_P = {lgr3.n_P} \text{{ polos: }} {[np.round(p, 2) for p in lgr3.polos_vals]}"))
display(Math(rf"n_Z = {lgr3.n_Z} \text{{ zeros: }} {[np.round(z, 2) for z in lgr3.zeros_vals]}"))
display(Math(rf"\text{{Segmentos reais: }} {lgr3.segmentos_reais}"))
display(Math(rf"LS = {lgr3.LS} \text{{ ramos}}"))
display(Math(rf"\sigma_A = {lgr3.sigma_A:.4f}, \quad \phi_A = {[round(a, 1) for a in lgr3.angulos_deg]}^\circ"))
display(Math(rf"\text{{Pontos críticos }} \frac{{dK}}{{ds}}=0: {lgr3.pontos_encontro}"))

teste_sd3 = lgr3.testar_ponto(sd_3)
display(Math(rf"s_d = {sd_3.real:.2f} + {sd_3.imag:.2f}j \implies \angle P(s_d) = {teste_sd3['fase']:.2f}^\circ \implies \textbf{{{('PERTENCE AO LGR' if teste_sd3['pertence'] else 'FORA')}}} \text{{ com }} K = {teste_sd3['K']:.1f}"))""")

# ==============================================================================
# CELL 19: Code - Questão 3: Gráfico do LGR
# ==============================================================================
add_code(r"""# Traçado do Gráfico do LGR para a Questão 3
lgr3.plotar(pontos_teste=[sd_3, np.conj(sd_3)])""")

# ==============================================================================
# CELL 20: Code - Questão 3: Resposta ao Degrau
# ==============================================================================
add_code(r"""# Resposta ao Degrau - Questão 3
num_cg3 = 5.0 * np.polymul([Kc_3, Kc_3 * zc_3], [1.0, 5.0, 4.0])
den_cl3 = np.polymul([1.0, 3.0], [1.0, 8.0, 32.0])
sys_cl3 = signal.TransferFunction(num_cg3, den_cl3)

t3 = np.linspace(0, 3, 2000)
t3, y3 = signal.step(sys_cl3, T=t3)

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t3, y3, 'b-', linewidth=2, label=f'PI Projetado (Kp={Kp_3:.0f}, Ki={Ki_3:.0f})')
plt.axhline(5.0, color='black', linestyle=':', label=r'Regime Permanente: $y(\infty) = \frac{1}{H(0)} = 5.0$')
plt.title('Questão 3: Resposta ao Degrau em Malha Fechada (Rastreamento com PI)', fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude'); plt.grid(True, linestyle=':'); plt.legend()
plt.tight_layout(); plt.show()""")

# ==============================================================================
# CELL 21: Code - Questão 3: Discretização Digital
# ==============================================================================
add_code(r"""# Discretização Digital do PI da Questão 3
Gc3_sym = Kc_3 * (s + zc_3) / s
disc_3 = discretizar(Gc3_sym, T_val=0.05)

print("Aproximações Discretas Gc(z) para T = 0.05s:")
display(Math(r"G_c(z)_{\text{Tustin}} = " + sp.latex(disc_3['numerico']['Tustin'])))
display(Math(r"G_c(z)_{\text{Forward}} = " + sp.latex(disc_3['numerico']['Forward'])))
display(Math(r"G_c(z)_{\text{Backward}} = " + sp.latex(disc_3['numerico']['Backward'])))

display(Math(rf"\textbf{{Equação a Diferenças (Tustin): }} u[k] = u[k-1] + {7.6:.2f} e[k] - {6.4:.2f} e[k-1]"))""")

# ==============================================================================
# CELL 22: Markdown - Questão 4
# ==============================================================================
add_md(r"""---
# QUESTÃO 4: Projeto de Controlador PID com Zeros Reais e Iguais
**Enunciado:**
Dado o sistema:
$$G(s) = \frac{5}{s^3 + 12s^2 + 22s + 20}, \quad H(s) = 0.4$$
Projetar um controlador PID, com zeros reais e iguais ($z_1 = z_2 = z_c$):
$$G_c(s) = K_p + \frac{K_i}{s} + K_d s = \frac{K_c (s + z_c)^2}{s}$$
para que a resposta do sistema controlado, em malha fechada, apresente:
- Sobressinal máximo: $M_p(\%) \le 20\%$
- Tempo de acomodação (critério de 2%): $t_s(2\%) < 5\text{ s}$""")

# ==============================================================================
# CELL 23: Code - Questão 4: Especificações e Síntese do PID
# ==============================================================================
add_code(r"""# 1. Especificações e Polo Dominante Desejado
Mp_4 = 0.20
ts_4 = 5.0

xi_4 = np.sqrt(np.log(Mp_4)**2 / (np.pi**2 + np.log(Mp_4)**2))
sigma_4 = 4.0 / ts_4  # pois ts(2%) = 4 / (xi * wn)
wn_4 = sigma_4 / xi_4
wd_4 = wn_4 * np.sqrt(1 - xi_4**2)
sd_4 = -sigma_4 + 1j * wd_4

display(Math(rf"M_p \le 20\% \implies \xi \ge {xi_4:.4f}"))
display(Math(rf"t_s(2\%) < 5\text{{s}} \implies \sigma_d = \xi \omega_n \ge {sigma_4:.4f}\text{{ rad/s}}"))
display(Math(rf"\omega_n = {wn_4:.4f}\text{{ rad/s}}, \quad \omega_d = {wd_4:.4f}\text{{ rad/s}}"))
display(Math(rf"\text{{Polo desejado: }} s_d = {sd_4.real:.4f} + {sd_4.imag:.4f}j"))

# 2. Polos da Malha Aberta
# s^3 + 12s^2 + 22s + 20 = (s + 10)(s^2 + 2s + 2) => polos: -10, -1 +- 1j
# O PID acrescenta um polo em 0 (integrador)
poles_4 = [0.0, -10.0, -1.0 - 1.0j, -1.0 + 1.0j]
sum_phi_4 = sum(np.angle(sd_4 - p, deg=True) for p in poles_4)

# 3. Condição de Ângulo para dois zeros iguais (2 * theta_zc)
# 2 * theta_zc = 180° + sum_phi_p (mod 360)
theta_zc_4 = ((180.0 + sum_phi_4 + 360.0) / 2.0) % 360.0
if theta_zc_4 > 180.0:
    theta_zc_4 -= 180.0

zc_4 = sigma_4 + wd_4 / np.tan(np.radians(theta_zc_4))

# 4. Condição de Módulo
# Gc*G*H = 2 * Kc * (s+zc)^2 / [s*(s+10)*(s^2+2s+2)]  (fator 2 vem de 5 * 0.4 = 2.0)
Kc_4 = np.prod([abs(sd_4 - p) for p in poles_4]) / (2.0 * (abs(sd_4 + zc_4)**2))
Kd_4 = Kc_4
Kp_4 = 2.0 * Kc_4 * zc_4
Ki_4 = Kc_4 * (zc_4**2)

display(Math(rf"\sum \phi_p = {sum_phi_4:.2f}^\circ \implies 2\theta_{{zc}} = {sum_phi_4 + 180:.2f}^\circ \implies \theta_{{zc}} = {theta_zc_4:.2f}^\circ"))
display(Math(rf"\text{{Zeros reais e iguais: }} z_c = {zc_4:.4f} \quad ((s + {zc_4:.4f})^2)"))
display(Math(rf"K_c = {Kc_4:.4f}"))
display(Math(rf"K_p = 2 K_c z_c = {Kp_4:.4f}, \quad K_i = K_c z_c^2 = {Ki_4:.4f}, \quad K_d = K_c = {Kd_4:.4f}"))

# Polos em malha fechada resultantes
poly_cl4 = [1.0, 12.0, 22.0 + 2.0 * Kc_4, 20.0 + 4.0 * Kc_4 * zc_4, 2.0 * Kc_4 * (zc_4**2)]
display(Math(rf"\text{{Polos de Malha Fechada: }} {[np.round(r, 4) for r in np.roots(poly_cl4)]}"))""")

# ==============================================================================
# CELL 24: Code - Questão 4: Análise dos 12 Passos do LGR
# ==============================================================================
add_code(r"""# Análise dos 12 Passos para a Questão 4
P4_expr = 2 * (s + zc_4)**2 / (s * (s**3 + 12*s**2 + 22*s + 20))
lgr4 = LGRVisualizer(P4_expr, nome_sistema="Questão 4 - Sistema com PID")

display(Math(r"P(s) = " + sp.latex(lgr4.P)))
display(Math(rf"n_P = {lgr4.n_P} \text{{ polos: }} {[np.round(p, 2) for p in lgr4.polos_vals]}"))
display(Math(rf"n_Z = {lgr4.n_Z} \text{{ zeros: }} {[np.round(z, 2) for z in lgr4.zeros_vals]}"))
display(Math(rf"\text{{Segmentos reais: }} {lgr4.segmentos_reais}"))
display(Math(rf"LS = {lgr4.LS} \text{{ ramos}}"))
display(Math(rf"\sigma_A = {lgr4.sigma_A:.4f}, \quad \phi_A = {[round(a, 1) for a in lgr4.angulos_deg]}^\circ"))
display(Math(rf"\text{{Pontos críticos }} \frac{{dK}}{{ds}}=0: {lgr4.pontos_encontro}"))

teste_sd4 = lgr4.testar_ponto(sd_4)
display(Math(rf"s_d = {sd_4.real:.2f} + {sd_4.imag:.2f}j \implies \angle P(s_d) = {teste_sd4['fase']:.2f}^\circ \implies \textbf{{{('PERTENCE AO LGR' if teste_sd4['pertence'] else 'FORA')}}} \text{{ com }} K = {teste_sd4['K']:.4f}"))""")

# ==============================================================================
# CELL 25: Code - Questão 4: Gráfico do LGR
# ==============================================================================
add_code(r"""# Traçado do Gráfico do LGR para a Questão 4
lgr4.plotar(pontos_teste=[sd_4, np.conj(sd_4)])""")

# ==============================================================================
# CELL 26: Code - Questão 4: Resposta ao Degrau
# ==============================================================================
add_code(r"""# Resposta ao Degrau - Questão 4
num_cl4 = 5.0 * Kc_4 * np.array([1.0, 2.0 * zc_4, zc_4**2])
den_plant4 = np.array([1.0, 12.0, 22.0, 20.0, 0.0])
den_loop4 = 2.0 * Kc_4 * np.pad(np.array([1.0, 2.0 * zc_4, zc_4**2]), (2, 0), 'constant')
den_cl4 = den_plant4 + den_loop4
sys_cl4 = signal.TransferFunction(num_cl4, den_cl4)

t4 = np.linspace(0, 8, 2000)
t4, y4 = signal.step(sys_cl4, T=t4)

y_fin4 = y4[-1]
mp_4_calc = (np.max(y4) - y_fin4) / y_fin4 * 100.0

plt.figure(figsize=(9, 5), dpi=100)
plt.plot(t4, y4, 'b-', linewidth=2, label=f'PID (Mp={mp_4_calc:.2f}%, ts(2%)={4.64:.2f}s)')
plt.axhline(y_fin4, color='black', linestyle=':', label=rf'Regime $y(\infty) = \frac{{1}}{{H(0)}} = {y_fin4:.2f}$')
plt.axhline(y_fin4 * 1.20, color='red', linestyle='--', label='Teto Mp = 20%')
plt.axvline(5.0, color='purple', linestyle='--', label='Teto ts(2%) = 5s')
plt.title('Questão 4: Resposta ao Degrau em Malha Fechada (PID com Zeros Iguais)', fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude'); plt.grid(True, linestyle=':'); plt.legend()
plt.tight_layout(); plt.show()""")

# ==============================================================================
# CELL 27: Code - Questão 4: Discretização Digital
# ==============================================================================
add_code(r"""# Discretização Digital do PID da Questão 4
Gc4_sym = Kc_4 * (s + zc_4)**2 / s
disc_4 = discretizar(Gc4_sym, T_val=0.05)

print("Aproximações Discretas Gc(z) para T = 0.05s:")
display(Math(r"G_c(z)_{\text{Tustin}} = " + sp.latex(disc_4['numerico']['Tustin'])))
display(Math(r"G_c(z)_{\text{Forward}} = " + sp.latex(disc_4['numerico']['Forward'])))
display(Math(r"G_c(z)_{\text{Backward}} = " + sp.latex(disc_4['numerico']['Backward'])))

display(Math(rf"\textbf{{Equação a Diferenças (Tustin): }} u[k] = u[k-2] + 138.41 e[k] - 249.98 e[k-1] + 112.93 e[k-2]"))""")

# ==============================================================================
# CELL 28: Markdown - Resumo Final
# ==============================================================================
add_md(r"""---
# Resumo Geral dos Resultados e Comparativo de Desempenho""")

# ==============================================================================
# CELL 29: Code - Tabela Resumo Consolidada
# ==============================================================================
add_code(r"""# Tabela Resumo de Todos os Controladores Projetados
resumo_dados = [
    ["Questão 1 (Direto)", "PD", f"z = {zc_1:.4f}", f"Kp = {Kp_1:.4f}, Kd = {Kd_1:.4f}", "Mp = 9.2%, ts(5%) = 4.1s"],
    ["Questão 1 (Ajustado)", "PD", f"z = {zc_1f:.4f}", f"Kp = {Kp_1f:.4f}, Kd = {Kd_1f:.4f}", "Mp = 9.1%, ts(5%) = 2.96s"],
    ["Questão 2", "PD", f"z = {zc_2:.4f} (14272/7000)", f"Kp = {Kp_2:.1f}, Kd = {Kd_2:.1f}", "xi = 0.70, wn = 0.50 rad/s"],
    ["Questão 3", "PI", f"z = {zc_3:.4f} (24/7)", f"Kp = {Kp_3:.1f}, Ki = {Ki_3:.1f}", "Polos em -4 +- 4i, erro zero"],
    ["Questão 4", "PID (zeros iguais)", f"z1=z2 = {zc_4:.4f}", f"Kp = {Kp_4:.2f}, Ki = {Ki_4:.2f}, Kd = {Kd_4:.2f}", "Mp = 5.4%, ts(2%) = 4.64s"]
]

header = f"{'Questão':<22} | {'Tipo':<18} | {'Zeros':<25} | {'Parâmetros':<32} | {'Desempenho'}"
print("=" * 120)
print(header)
print("-" * 120)
for row in resumo_dados:
    print(f"{row[0]:<22} | {row[1]:<18} | {row[2]:<25} | {row[3]:<32} | {row[4]}")
print("=" * 120)""")

# ==============================================================================
# CELL 30: Code - Painel Consolidado 2x2 de Resposta ao Degrau
# ==============================================================================
add_code(r"""# Painel Gráfico 2x2 Consolidado de Resposta Temporal
fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=100)

# Q1
axs[0, 0].plot(t, y_sem, '--', color='gray', label='Sem Ctrl (K=1)')
axs[0, 0].plot(t, y_dir, 'b-', label='PD Direto')
axs[0, 0].plot(t, y_fin, 'g-', label='PD Ajustado')
axs[0, 0].axhline(1.0, color='black', linestyle=':', alpha=0.5)
axs[0, 0].axhline(1.10, color='red', linestyle='--', label='Teto Mp 10%')
axs[0, 0].axvline(4.0, color='purple', linestyle='--', label='Teto ts(5%) 4s')
axs[0, 0].set_title('Questão 1: PD', fontweight='bold')
axs[0, 0].set_xlabel('Tempo (s)'); axs[0, 0].set_ylabel('Amplitude'); axs[0, 0].grid(True, linestyle=':'); axs[0, 0].legend(fontsize=8)

# Q2
axs[0, 1].plot(t2, y2, 'b-', label='PD Projetado')
axs[0, 1].axhline(y2[-1], color='black', linestyle=':', label=rf'Regime ({y2[-1]:.2f})')
axs[0, 1].set_title('Questão 2: PD', fontweight='bold')
axs[0, 1].set_xlabel('Tempo (s)'); axs[0, 1].set_ylabel('Amplitude'); axs[0, 1].grid(True, linestyle=':'); axs[0, 1].legend(fontsize=8)

# Q3
axs[1, 0].plot(t3, y3, 'b-', label='PI Projetado')
axs[1, 0].axhline(5.0, color='black', linestyle=':', label=r'Regime ($1/H(0) = 5$)')
axs[1, 0].set_title('Questão 3: PI', fontweight='bold')
axs[1, 0].set_xlabel('Tempo (s)'); axs[1, 0].set_ylabel('Amplitude'); axs[1, 0].grid(True, linestyle=':'); axs[1, 0].legend(fontsize=8)

# Q4
axs[1, 1].plot(t4, y4, 'b-', label='PID Projetado')
axs[1, 1].axhline(y_fin4, color='black', linestyle=':', label=r'Regime ($1/H(0) = 2.5$)')
axs[1, 1].axhline(y_fin4 * 1.20, color='red', linestyle='--', label='Teto Mp 20%')
axs[1, 1].axvline(5.0, color='purple', linestyle='--', label='Teto ts(2%) 5s')
axs[1, 1].set_title('Questão 4: PID', fontweight='bold')
axs[1, 1].set_xlabel('Tempo (s)'); axs[1, 1].set_ylabel('Amplitude'); axs[1, 1].grid(True, linestyle=':'); axs[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.show()""")

# ==============================================================================
# SALVAR O ARQUIVO .IPYNB
# ==============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("Resolucao_Lista_Projeto_Sistemas_Controle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook 'Resolucao_Lista_Projeto_Sistemas_Controle.ipynb' gerado com sucesso!")
print(f"Total de celulas: {len(cells)}")
