import numpy as np
import itertools
import random
import re


DIMENSION_INICIAL = 25
VARIABLES_INICIALES = [f"x{i}" for i in range(DIMENSION_INICIAL)]
SAMPLE_SIZE = min(2000, pow(2,DIMENSION_INICIAL))    # muestras aleatorias para evaluación y entrenamiento
HIDDEN_SIZE = min(64, pow(2,DIMENSION_INICIAL))      # tamaño de la capa oculta de la red neuronal
ELITISMO = 5

def sigmoid(z):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    """Derivada de la función sigmoide."""
    s = sigmoid(z)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """Propagación hacia adelante."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)  # Para clasificación binaria
        return self.A2
    
    def compute_loss(self, Y, A2):
        """Cálculo del costo (entropía cruzada)."""
        m = Y.shape[0]
        loss = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        return loss
    
    def backward(self, X, Y, learning_rate):
        """Propagación hacia atrás y actualización de parámetros."""
        m = X.shape[0]
        dZ2 = self.A2 - Y                     # Derivada del costo respecto a Z2
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Actualización de parámetros
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, Y, epochs, learning_rate, print_loss=False):
        """Entrenamiento de la red neuronal."""
        for i in range(epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            self.backward(X, Y, learning_rate)


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


def fitness (ind, X, Y, size_weight=0.01, abstraction_weight=0.05):
    puntuacion = 0
    dictionary = {}
    casos = 0
    for j in range(SAMPLE_SIZE):
        key = tuple(X[j])
        if key in dictionary:
            if dictionary[key] != Y[j]:
                puntuacion += 1
        else:
            casos += 1
            dictionary[key] = Y[j]
    for expr in ind:
        puntuacion += size_weight * len(expr)
    puntuacion += abstraction_weight * casos
    return puntuacion




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




if __name__ == "__main__":
    
    
    #Conjuntos X e Y
    if(SAMPLE_SIZE == pow(2,DIMENSION_INICIAL)):
        X = np.array([list(c) for c in itertools.product([0, 1], repeat=DIMENSION_INICIAL)])
    else:
        X = np.array([tuple(random.randint(0,1) for _ in range(DIMENSION_INICIAL)) for _ in range(SAMPLE_SIZE)])
    Y = np.array([[f6(*s)] for s in X])

    
    variables = VARIABLES_INICIALES
    max_dim = len(variables) + 2
    min_dim = 1
    current_dim = len(variables)
    initial_prob_inc = 0.25        # probabilidad inicial de aumentar dimensión
    step = 0
    
    
    while True:
        # Probabilidad de aumentar que decae con el número de iteración
        prob_increase = initial_prob_inc / (step + 1)
        prob_increase = min(prob_increase, 1.0)
        
        # Decidir si subimos o bajamos la dimensión
        if random.random() < prob_increase and current_dim < max_dim:
            current_dim += 1
        else:
            current_dim -= 1
            # Asegurarnos de no pasarnos de los límites
        current_dim = max(min_dim, min(current_dim, max_dim))
        dimension = current_dim
        
        lista_nuevas_combinaciones = ([generate_logical_combinations(variables, dimension) for _ in range(10000)])
        X2 = np.array([list(evaluate_expressions(c, X)) for c in lista_nuevas_combinaciones])    
            
        puntuaciones = []
        for i in range(10000):
            puntuacion = fitness(lista_nuevas_combinaciones[i],X2[i], Y)
            puntuaciones.append(puntuacion)
            
        index_mejores = sorted(range(len(puntuaciones)), key=lambda i: puntuaciones[i])[:ELITISMO]
        index_mejor = 0
        best_accuracy = 0
        for k in range(ELITISMO): 
            
            # Seleccionamos la mejor combinación y convertimos a array numérico
            X1 = np.array(X2[index_mejores[k]], dtype=np.float32)
            nn = NeuralNetwork(input_size=dimension, hidden_size=HIDDEN_SIZE, output_size=1)
            
            nn.train(X1, Y, epochs=10000, learning_rate=1.0, print_loss=True)
        
            # Fase de test: evaluamos la red sobre el mismo conjunto de datos
            predictions = nn.forward(X1)
            # Convertir las salidas a 0 o 1 utilizando un umbral de 0.5
            predicted_labels = (predictions > 0.5).astype(int)
            # Cálculo de la precisión
            accuracy = np.mean(predicted_labels == Y)
            print(lista_nuevas_combinaciones[index_mejores[k]])
            print("Exactitud "+ str(k) + ": " +str(accuracy))
            if accuracy > best_accuracy:
                index_mejor = k
                best_accuracy = accuracy
        print("------------------------------------------------------------")
        step+=1
        
        variables = lista_nuevas_combinaciones[index_mejores[index_mejor]]
        print(variables)
        # Si ya hemos llegado a la dimensión mínima, salimos
        if current_dim == min_dim:
            break
    
    print("EXPRESIÓN FINAL: ")
    print(variables)