# Relatório - Lista 1: Modelagem Geométrica e Animação

## 1. Operadores Afins Notórios

### 1.1 Rotação em torno da reta s = {(x,y,z) ∈ R³ | x=2 e y=1}

**Resposta:** A reta s é paralela ao eixo z passando por (2,1,0). A matriz homogênea 4×4 é:

```
H(θ) = [cos(θ)  -sin(θ)   0   2-2cos(θ)+sin(θ)]
       [sin(θ)   cos(θ)   0   1-cos(θ)-2sin(θ)]
       [  0        0      1          0        ]
       [  0        0      0          1        ]
```

Implementada em `1.1.py`, com animação de um quadrado girando suavemente em torno dessa reta.

---

### 1.2 Reflexão no plano C

**Resposta:** O plano C passa por P₀=(0,1,0) com vetores diretores v=(-2,4,-2) e w=(-1,-1,1).

Normal: **n** = v × w = (2,4,2) (simplificado: (1,2,1))

Equação: x + 2y + z = 2

**Operador afim:** x' = Ax + t, onde:

- A = I - 2(n·nᵀ)/||n||²
- t = 2c·n/||n||²

```
A = [1/3   -2/3   -1/3]     t = [2/3]
    [-2/3  -1/3   -2/3]         [4/3]
    [-1/3  -2/3   1/3 ]         [2/3]
```

Implementada em `1.2.py` com animação interpolada da reflexão.

---

### 1.3 Rotação em torno de D com translação axial

**Resposta:** Reta D = {(-t, 1-t, t)} tem direção u = (-1,-1,1)/√3.

A matriz usa Rodrigues para rotação θ mais translação axial de magnitude (2/π)θ:

```
H(θ) = [    R(θ)    |  P₀ + s·u - R·P₀ ]
       [  0  0  0   |        1         ]
```

Onde R é a rotação de Rodrigues e s = (2/π)θ.

Implementada em `1.3.py` com animação helicoidal de um quadrado.

---

## 2. Simulação de Movimento

### 2.1 Movimento circular de A para B

**Resposta:** Partícula move-se de A=(2,-2,-3) para B=(2,1,0) em arco circular centrado em C=(0,-1,-1).

- Vetores: CA = (2,-1,-2), CB = (2,2,1)
- Ângulo ACB ≈ 90° = π/2
- Eixo: u = (CA × CB) normalizado = (3,6,6)/9 = (1,2,2)/3
- Arcos de 30°: k = 3 iterações

**Matriz:** H₃₀ = T(C)·R(30°,u)·T(-C)

**Transformação total:** H_total = (H₃₀)³

Implementada em `2.1.py` com animação do arco circular em 3 etapas de 30°.

---

### 2.2 Composição: Rotação + Escala + Translação

**Resposta:** Sequência de transformações:

1. **Rotação 30° em torno da reta** passando por P₀=(-1,1,0) com direção u=(1,-1,1)/√3:

   - H_rot = T(P₀)·R(30°,u)·T(-P₀)

2. **Escala:** S = diag(3, -2, 0.5, 1)

3. **Translação:** T = [I | (1,-2,-3)]

**Matriz composta:** H_total = T·S·H_rot

Implementada em `2.2.py` com animação em 3 fases: rotação → escala → translação.

---

### 2.3 Reflexão + Rotação

**Resposta:** Composição de:

1. **Reflexão no plano x-y=1:**

   - n = (1,-1,0), c = 1
   - H_ref com A = I - 2(n·nᵀ)/||n||² e t = 2c·n/||n||²

2. **Rotação 30° anti-horária em torno de (t,0,-t) com direção (1,0,-1):**
   - H_rot = R(30°, (1,0,-1))

**Matriz composta:** H_total = H_rot · H_ref

Implementada em `3.1.py` com animação interpolada em 2 fases.

---

## 3. Pião Girante

**Resposta:** Pião com bico em (1,2,0) realiza dois movimentos compostos:

1. **Rotação própria:** 4 voltas em torno do eixo r = {(1+q, 2-q, 0)} (direção (1,-1,0))

   - ω_spin = 8π/T rad/s

2. **Precessão:** eixo r gira em torno de s = {x=2, y=1} (eixo vertical)
   - ω_r = 2π/T rad/s

**Matrizes parametrizadas:**

- H_precessão(t) = R_s(ω_r·t) - rotação do eixo r em torno de s
- H_spin(t) = R_r(ω_spin·t) - rotação do pião em torno de r

**Composição:** H_total(t) = H_precessão(t) · H_spin(t)

Implementada em `3.2.py` com T=4s, mostrando o pião girando e precessionando simultaneamente.

---

## 4. A Serpente Dimensional Isneique

### 4.1 Movimento espiral com reflexões

**Resposta:** Isneique move-se em espiral em torno do eixo D = {(-t, 1-t, t)} com:

- Direção: u = (-1,-1,1)/√3
- Rotação: ω = 2π/T
- Translação axial: 2 unidades/volta → shift = (1/π)·φ(t)

**Planos limítrofes:**

- A: -2x + y - z = 1
- B: y + z = 1
- C (reflexão): (1,2,1)·x = 2

**Operador espiral:** H_spiral(t) = T(shift·u) · R_D(ωt)

**Reflexão:** H_C = reflexão no plano C (aplicada quando cruza A ou B)

**Lógica:**

```
Se cruza plano A ou B:
    aplicar H_C (reflexão)
Continuar movimento espiral H_spiral(t)
```

Implementada em `4.1.py` com detecção de cruzamento e aplicação automática de reflexão.

---

### 4.2 Planos A e B rotativos

**Resposta:** Se os planos A e B começam a rotacionar em torno de D com velocidade angular ω:

**Adaptações necessárias:**

1. **Planos dinâmicos:** A(t) e B(t) são obtidos rotacionando as normais originais:

   - n_A(t) = R_D(ωt)·n_A₀ (sentido horário)
   - n_B(t) = R_D(-ωt)·n_B₀ (sentido anti-horário)
   - Constantes c_A e c_B se ajustam conforme o ponto na reta D

2. **Detecção de cruzamento:** Em cada frame, avaliar se a serpente cruza os planos A(t) ou B(t) usando o sinal de n(t)·p - c(t)

3. **Reflexão dinâmica:** Ao cruzar, aplicar H_C (plano C permanece fixo) e continuar movimento espiral

4. **Frequência de cruzamentos:** Como os planos giram com mesma ω do movimento espiral, os cruzamentos ocorrem em padrões periódicos complexos

Implementada em `4.1.2.py` com planos A e B animados rotacionando em sentidos opostos.

---


**Arquivos:** `1.1.py`, `1.2.py`, `1.3.py`, `2.1.py`, `2.2.py`, `3.1.py`, `3.2.py`, `4.1.py`, `4.1.2.py`
