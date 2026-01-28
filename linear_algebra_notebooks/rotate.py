import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider

'''
Rotação de Imagens Usando Operações Vetorizadas


Utiliza conceitos básicos de álgebra linear e manipulação de arrays para rotacionar imagens em torno do centro.
Centrado em torno da função `rotacao_vetorizada`, que aplica uma transformação de rotação 2D usando matrizes e operações vetorizadas do Numpy.
Gera diversas imagens rotacionadas e permite a visualização interativa com um slider.
Conceitos usados:
- Algebra Linear: Matrizes de rotação, transformações lineares.
- Numpy: Manipulação eficiente de arrays, operações vetorizadas.
- Matplotlib: Visualização de imagens, criação de sliders interativos.
'''




imagem = Image.open(r"lula molusco.jpg")
imagem_array = np.array(imagem)
black_white = imagem_array.mean(axis=2)

rows, cols = black_white.shape


def get_image(angle_deg, images):
    """Converte ângulo em graus radianos e mapeiapara índice e retorna a imagem correspondente."""
    angle_rad = angle_deg * np.pi / 180
    index = int(angle_rad / (2 * np.pi) * len(images)) % len(images)
    return images[index]


def rotacao_vetorizada(black_white, theta):
    """
    Rotaciona uma imagem ao redor do seu centro usando operações vetorizadas.
    
    Estratégia:
    1. Associa coordenadas (y, x) a cada pixel da imagem de destino
    2. Muda o referencial para o centro da imagem (origem no centro)
    3. Aplica rotação inversa usando matriz de rotação 2D
    4. Retorna ao referencial original
    5. Mapeia coordenadas float para inteiros (nearest neighbor)
    6. Filtra coordenadas válidas (dentro dos limites da imagem)
    7. Atribui valores dos pixels correspondentes da imagem original
    
    Matemática:
    - Matriz de rotação inversa (backward mapping):
      R^(-1)(θ) = [cos(θ)   sin(θ) ]
                  [-sin(θ)  cos(θ) ]
    
    - Transformação: p_src = R^(-1) @ (p_dest - center) + center
    
    Manipulação de dados:
    - Coordenadas são planificadas como vetores linha (1D) para operações vetorizadas
    - Operação matricial aplicada em batch: 2×N matriz onde N = rows × cols
    - ravel() mantém alinhamento entre mask e pixels (ordem row-major)
    
    Parameters:
    -----------
    black_white : np.ndarray
        Imagem em escala de cinza (2D array)
    theta : float
        Ângulo de rotação em radianos
    
    Returns:
    --------
    np.ndarray
        Imagem rotacionada (2D array, mesmas dimensões da entrada)
        Pixels inválidos (fora dos limites) ficam pretos (0)
    """
    # Matriz de rotação inversa (backward mapping para evitar buracos na imagem)
    c, s = np.cos(theta), np.sin(theta)
    R_inv = np.array([[c, s], [-s, c]])

    # Gera coordenadas (y, x) para cada pixel de destino
    # Planificadas em ordem row-major: [(0,0), (0,1), ..., (0,cols-1), (1,0), ...]
    yy, xx = np.indices((rows, cols))
    coords_dest = np.stack([yy.ravel(), xx.ravel()])  # Shape: (2, rows*cols)

    # Muda referencial: centro da imagem vira origem (0, 0)
    center = np.array([[rows / 2], [cols / 2]])
    coords_dest_centered = coords_dest - center

    # Aplica rotação inversa: encontra de onde cada pixel destino veio na imagem original
    coords_src_centered = R_inv @ coords_dest_centered

    # Retorna ao referencial original e mapeia para inteiros (nearest neighbor)
    coords_src = coords_src_centered + center
    coords_src = np.rint(coords_src).astype(int)  # rint + astype = mapping para coordenadas inteiras

    src_rows, src_cols = coords_src[0, :], coords_src[1, :]

    # Cria máscara: filtra apenas coordenadas dentro das dimensões da imagem original
    # Evita IndexError e wrapping negativo de índices
    mask = (src_rows >= 0) & (src_rows < rows) & (src_cols >= 0) & (src_cols < cols)

    # Atribui valores dos pixels válidos
    # ravel() planifica img_rotacionada na mesma ordem que coords_dest foi criado
    # Mantém correspondência: posição i no vetor = pixel (i//cols, i%cols) na matriz 2D
    img_rotacionada = np.zeros((rows, cols), dtype=black_white.dtype)
    img_rotacionada.ravel()[mask] = black_white[src_rows[mask], src_cols[mask]]
    
    return img_rotacionada


imagens = [rotacao_vetorizada(black_white, theta) for theta in np.linspace(0, 2 * np.pi, 100)]

fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(imagens[0], cmap='gray', interpolation='bilinear')


fig.subplots_adjust(bottom=0.15, top=0.95)
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

slider = Slider(
    ax=ax_slider,
    label='Angle [°]',
    valmin=0,
    valmax=360,
    valinit=0,
    color="#3134EA",
    track_color='#E0E0E0'
)


def update(angle):
    """Atualiza a imagem exibida com base no ângulo do slider."""
    im.set_data(get_image(angle, imagens))
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
