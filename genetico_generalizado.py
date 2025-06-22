# -*- coding: utf-8 -*-
"""
Algoritmo genético adaptado a 50 variables usando muestreo en la evaluación de fitness e integración de red neuronal
"""
import numpy as np
import random
import re
import itertools
from collections import defaultdict, Counter

from sympy import symbols, simplify_logic
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)



DIMENSION_INICIAL = 4
VARIABLES_INICIALES = [f"x{i}" for i in range(DIMENSION_INICIAL)]
SAMPLE_SIZE = min(2000, pow(2,DIMENSION_INICIAL))    # muestras aleatorias para evaluación y entrenamiento
HIDDEN_SIZE = min(64, pow(2,DIMENSION_INICIAL))      # tamaño de la capa oculta de la red neuronal

# -------- factor-común -----------------------------
FACTOR_FREQ        = 5       # se aplica cada K generaciones
FACTOR_TOP_N       = 10      # nº de mejores individuos que se pulen
FACTOR_MAX_LITS    = 2       # conjunción de ≤ 2 literales
FACTOR_MIN_GAIN    = 8       # confusiones mínimas que debe resolver
FACTOR_SUBSAMPLE   = 512     # filas máx. para estimar confusiones

# --- Funciones de activación y red neuronal ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        return -1/m * np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))

    def backward(self, X, Y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, Y, epochs, learning_rate, print_loss=False):
        for i in range(epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            self.backward(X, Y, learning_rate)
            if print_loss and i % 1000 == 0:
                print(f"Época {i}, Costo: {loss:.4f}")





def simplify_logical_expression(expr_str, form='dnf'):
    """
    Simplifica una expresión booleana en sintaxis Python (and, or, not, xor)
    a DNF o CNF usando Sympy, evaluando con operadores bitwise.

    Args:
        expr_str (str): Expresión booleana en sintaxis Python.
        form (str): 'dnf' o 'cnf'.
    Returns:
        str: Expresión simplificada en sintaxis Python.
    """
    # 1) Crear símbolos y mapa local
    syms = symbols(' '.join(VARIABLES_INICIALES))
    sym_map = dict(zip(VARIABLES_INICIALES, syms))

    # 2) Convertir expr_str a cadena y reemplazar keywords por bitwise
    s = str(expr_str)
    s = re.sub(r'\bnot\b', '~', s)
    s = re.sub(r'\band\b', '&', s)
    s = re.sub(r'\bor\b', '|', s)
    s = re.sub(r'\bxor\b', '^', s)

    # 3) Evaluar directamente con eval usando símbolos
    try:
        expr = eval(s, {}, sym_map)
    except Exception as e:
        raise ValueError(f"Error al evaluar expresión: {e}")

    # 4) Simplificar la expresión booleana
    simplified = simplify_logic(expr, form=form)

    # 5) Reconstruir sintaxis Python legible
    out = str(simplified)
    out = out.replace('~', 'not ')
    out = out.replace('&', ' and ')
    out = out.replace('|', ' or ')
    out = out.replace('^', ' xor ')
    out = re.sub(r'\s+', ' ', out).strip()
    return out








def f1(*args):
    return random.randint(0, 1)

def f2(*args):
    vals = np.array(args)
    mid = len(vals) //2
    part1 = np.bitwise_or.reduce(vals[:mid])
    part2 = np.bitwise_and.reduce(vals[mid:])
    return int(part1 ^ part2)

def f3(*args):
    vals = np.array(args)
    mid = len(vals) // 2       
    part1 = np.bitwise_xor.reduce(vals[:mid])
    part2 = np.bitwise_or.reduce(vals[mid:])
    return int((~part1) & part2)

def f4(*args):

    vals = np.array(args, dtype=np.bool_)
    n      = len(vals)
    t      = n // 3 or 1                # al menos 1 elemento por tercio

    # 1) Señales base (cada una aparece varias veces más abajo)
    p = np.bitwise_or.reduce(vals[:t])          # OR   del 1er tercio
    q = np.bitwise_xor.reduce(vals[t:2*t])      # XOR  del 2º tercio
    r = np.bitwise_and.reduce(vals[2*t:])       # AND  del 3º tercio

    # 2) Mezclas intermedias —-ya no tan “limpias”-—
    a = (p & q) ^ (~q | r)              # p y q en positivo y negado
    b = (~p & r) | (q ^ r)              # r aparece con y sin negación
    c = (p | ~r) & (q | p)              # p, q, r de nuevo en varios modos

    # 3) Expresión final: cada sub-bloque vuelve a usar p, q, r
    result = (
        ((a & b) ^ c)          |        # a, b, c juntos
        (a & r)                |        # r reaparece
        ((~b) & p)             ^        # p y neg(b)
        (c & q)                         # q mezclado con c
    )

    # Devuelve uint64 “limpio”
    return int(result)


def f5(*args):
    vals = np.array(args, dtype=np.bool_)
    mid  = len(vals) // 2 or 1            # al menos un elemento por mitad

    # Señales base
    a = np.bitwise_or.reduce(vals[:mid])   # OR   sobre la 1.ª mitad
    b = np.bitwise_xor.reduce(vals[mid:])  # XOR  sobre la 2.ª mitad

    # Mezclas intermedias (cada variable aparece varias veces)
    m1 = (a & ~b) | (b ^ a)               # usa a, ~b, b, a
    m2 = (~a | b) ^ (a & b)               # usa ~a, b, a, b

    # Expresión final: repite a, b, m1, m2 afirmados y negados
    res = (
        (m1 & m2) ^                       # m1, m2
        ((~m1) | (a & b))                 # ~m1, a, b
    )

    # Devuelve uint64 normalizado
    return int(res)


def f6(*args):
    vals = np.array(args, dtype=np.bool_)

    # 1) División en tres partes de tamaño ≈ igual (siempre ≥1 elemento)
    part1, part2, part3 = np.array_split(vals, 3)

    # 2) Señales base (comparten variables entre sí de forma implícita)
    x = np.bitwise_or.reduce(part1)      # OR   grupo 1
    y = np.bitwise_xor.reduce(part2)     # XOR  grupo 2
    z = np.bitwise_and.reduce(part3)     # AND  grupo 3

    # 3) Mezclas intermedias (cada variable aparece varias veces, negada y sin negar)
    p = (x ^ y) & (~z | x)
    q = (y | z) ^ (x & ~y)
    r = (~x & z) | (y ^ z)

    # 4) Expresión final: se cruzan p, q, r y sus negaciones
    res = (p ^ q) | (r & p) ^ (~q & r)

    # 5) Devuelve 0 ó 1
    return int(res)







def generate_logical_combinations(variables, max_combinations):
    """
    Genera expresiones lógicas aleatorias a partir de una lista de variables.
    
    Args:
        variables (list): Lista de variables (por ejemplo, ['a', 'b', 'c', 'd', 'e']).
        max_combinations (int): Número de expresiones lógicas únicas a generar.
        
    Returns:
        list: Lista de expresiones lógicas (strings) generadas.
    """
    max_vars = len(variables)//2
    if len(variables) < 2:
        raise ValueError("El conjunto debe contener al menos dos variables para generar combinaciones.")

    logical_combinations = set()  # Usamos un set para evitar duplicados
    max_attempts = max_combinations * 10
    attempts = 0
        
        
    while len(logical_combinations) < max_combinations and attempts < max_attempts:

        num_vars = random.randint(1, max_vars)

        # Seleccionar 'num_vars' variables de forma aleatoria
        selected_vars = random.choices(variables, k = num_vars)

        # Construir la combinación lógica
        combination = ""
        for i, var in enumerate(selected_vars):
            if i > 0:
                operator = random.choice(["and", "or", "xor"])
                combination += f" {operator} "
            # Con un 30% de probabilidad, anteponer NOT a la variable
            if random.random() < 0.3:
                combination += f"not {var}"
            else:
                combination += var

        # Añadir paréntesis a toda la combinación
        combination = f"({combination})"
        logical_combinations.add(combination)
        attempts += 1

    return list(logical_combinations)

def n(x):
    """Función NOT para 0 y 1: retorna 1 si x es 0, y 0 si x es 1."""
    return 1 - x


def evaluate_expressions(expressions, X):
    variables = VARIABLES_INICIALES
    results = []
    
    # Preprocesamos cada expresión para transformarla a una sintaxis evaluable por Python.
    processed_expressions = []
    for expr in expressions:
        # Reemplazar "not <variable>" por "n(<variable>)" usando regex
        expr_proc = re.sub(r'\bnot\s+([a-zA-Z]\w*)\b', r'n(\1)', expr)
        # Reemplazar operadores lógicos por bit a bit:
        expr_proc = expr_proc.replace("xor", "^")
        expr_proc = expr_proc.replace("and", "&")
        expr_proc = expr_proc.replace("or", "|")
        processed_expressions.append(expr_proc)
    
    # Evaluar para cada combinación de 0 y 1 del sample
    for values in X:
        context = dict(zip(variables, values))
        # Incluir la función n en el contexto para poder usarla en eval
        context['n'] = n
        eval_results = []
        for expr in processed_expressions:
            try:
                result = eval(expr, {}, context)
                # Asegurarse de que el resultado sea 0 o 1 (por ejemplo, True/False se convierten)
                result = int(result)
            except Exception :
                result = 0
            eval_results.append(result)
        results.append(eval_results)

    return results


def fitness(ind, X, Y, dimension_weight, size_weight, conservativity_weight):
    score = 0
    dictionary = {}
    casos = 0
    for j in range(SAMPLE_SIZE):
        key = tuple(X[j])
        if key in dictionary:
            if dictionary[key] != Y[j]:
                score += conservativity_weight 
        else:
            casos += 1
            dictionary[key] = Y[j]

    for expr in ind:
        score += size_weight * len(expr)

    score += size_weight * casos
    score += dimension_weight * len(ind)
    return score




# --- Operadores genéticos ---
def tournament_selection(pop, fits, k=3):
    sel = random.sample(list(zip(pop, fits)), k)
    return min(sel, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    child = []

    for expr in parent1:
        if random.random() < 0.5:
            child.append(expr)
    for expr in parent2:
        if random.random() < 0.5:
            child.append(expr)

    if not child:
        # combinamos ambos padres en un pool
        pool = parent1 + parent2
        if pool:
            child.append(random.choice(pool))
        else:
            # ambos padres vacíos: generamos una expr nueva
            child.append(generate_logical_combinations(VARIABLES_INICIALES, 1)[0])

    # eliminamos duplicados y devolvemos
    return list(set(child))

def mutation(ind, inc_rate=0.3, dec_rate=0.3, comb_rate = 0.3):
    new_ind = ind.copy()
    if random.random() < inc_rate:
        new_ind.append(generate_logical_combinations(VARIABLES_INICIALES, 1)[0])
    if len(new_ind) > 1 and random.random() < dec_rate:
        del new_ind[random.randrange(len(new_ind))]
    if len(new_ind) > 1 and random.random() < dec_rate:
        base = new_ind.copy()
        new_ind = generate_logical_combinations(base, random.randint(max(len(base) - 2, 1), len(base)+2))

    return new_ind

# --- Algoritmo genético principal ---
def genetic_algorithm(X, Y,
                      population_size=300, num_generations=100,
                      min_weight=SAMPLE_SIZE/2, max_weight=SAMPLE_SIZE - 1, 
                      total_weight=SAMPLE_SIZE):
    
    max_dim = len(VARIABLES_INICIALES) + 2
    min_dim = 1
    population = [generate_logical_combinations(VARIABLES_INICIALES, random.randint(min_dim, max_dim))
                  for _ in range(population_size)]

    best_ind, best_fit = None, float('inf')
    
    cont = 0
    gen = 0
    while gen < num_generations:
        frac = gen / num_generations
        dimension_weight = min_weight + (max_weight - min_weight) * frac
        
        other_weight = total_weight - dimension_weight
        conservativity_weight = 0.99 * other_weight
        size_weight = 0.01 * other_weight
        
        fits = [fitness(ind,evaluate_expressions(ind, X), Y, dimension_weight, size_weight, conservativity_weight)
                for ind in population]

      
        
        
        idx = np.argmin(fits)
        best_ind, best_fit = population[idx], fits[idx]
        print(f"Gen {gen + cont}: Best fitness = {best_fit:.2f}, Ind = {best_ind} ")

        new_pop = []
        elite_n = max(1, population_size//10)
        elite_idx = np.argsort(fits)[:elite_n]
        counter_better = 0
        total_offspring = 0
        for i in elite_idx:
            new_pop.append(population[i])
        while len(new_pop) < population_size:
            p1 = tournament_selection(population, fits)
            p2 = tournament_selection(population, fits)
            
            child = crossover(p1, p2)
            
            child = mutation(child)
            new_pop.append(child)
        population = new_pop
        gen +=1

        if gen == num_generations -1 and len(best_ind) != 1:
            gen-=1
            cont+=1     
        print(dimension_weight, size_weight, conservativity_weight)
    return best_ind, best_fit


# --- Ejecución de ejemplo con red neuronal ---
if __name__ == "__main__":


    # Preparar datos para entrenamiento de la red neuronal
    if(SAMPLE_SIZE == pow(2,DIMENSION_INICIAL)):
        X = np.array([list(c) for c in itertools.product([0, 1], repeat=DIMENSION_INICIAL)])
    else:
        X = np.array([tuple(random.randint(0,1) for _ in range(DIMENSION_INICIAL)) for _ in range(SAMPLE_SIZE)])
        
    Y = np.array([[f1(*s)] for s in X])
    print(Y)
    
    best, score = genetic_algorithm(X, Y)
    print("\nMejor individuo:", simplify_logical_expression(best[0]))
    print("Fitness final:", score)
    
    
    print(best)
    print(X)
    
    X_new = np.array(evaluate_expressions(best, X))
    print(X_new)
    print(Y)
    nn = NeuralNetwork(input_size=X_new.shape[1], hidden_size=HIDDEN_SIZE, output_size=1)
    nn.train(X_new, Y, epochs=10000, learning_rate=1.0, print_loss=True)
    preds = nn.forward(X_new)
    labels = (preds > 0.5).astype(int)
    print("Exactitud:", np.mean(labels == Y))