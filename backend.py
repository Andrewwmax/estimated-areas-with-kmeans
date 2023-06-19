# USE
# python backend.py --image images/test.png --clusters 4

import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

def centroid_histogram(clt):
    """
    Calcula e retorna o histograma dos centróides do cluster fornecido.

    Parâmetros:
    - clt: Um objeto de clustering k-means contendo os rótulos dos clusters.

    Retorna:
    - Um array numpy representando o histograma dos centróides do cluster.
    """

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    """
    Dada um histograma e seus centróides correspondentes, essa função plota as cores representadas pelos centróides
    em uma imagem de 400px x 200px. A imagem resultante é retornada como um array numpy tridimensional com dtype uint8.
    
    Parâmetros:
    hist (list): Uma lista de floats representando a frequência relativa de cada cor na imagem.
    centroids (list): Uma lista de arrays numpy com shape (3,) representando os valores RGB de cada centróide.
    
    Retorna:
    numpy.ndarray: Um array numpy tridimensional de dtype uint8 com shape (400, 200, 3) representando as cores plotadas.
    """

    bar = np.zeros((400, 200, 3), dtype = "uint8")
    startX = 0

    for i in range(len(hist)):
        print("COR: {0:}, \tArea: {1:.3f}".format(np.uint8(centroids[i]), hist[i] * 100))

    for (percent, color) in zip(hist, centroids):
        # endX = startX + (percent * 50)
        i = 0
        tam = 400 / len(hist)
        distText = tam / 2

        endX = startX + (tam)
        cv2.rectangle(bar, (200, int(endX)), (0, int(startX)), color.astype("uint8").tolist(), -1)
        
        text = '{0:.2f}'.format(percent * 100)
        
        cv2.putText(bar, text, (75, 5 + int(distText + startX)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(bar, text, (75, 5 + int(distText + startX)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        
        i = i + 1
        startX = endX

    return bar

def show_images(images, cols = 1, titles = None):
    # Thank's for https://gist.github.com/soply
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    
    n_images = len(images)
    
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    
    fig = plt.figure()
    
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, int(np.ceil(n_images/cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image),plt.xticks([]), plt.yticks([])
        a.set_title(title)
    
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # on Windows!
    
    plt.savefig('plots/plt_' + filename.split('/')[1])
    # plt.show()

def mainMenu():
    # Este código é para quantização de imagens usando clustering k-means.
    # O usuário insere o caminho para uma imagem e o número de clusters a serem usados para a quantização.
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
    args = vars(ap.parse_args())

    global filename
    # filename = 'images/test.png'
    filename = args["image"]
    
    # kCluster = 5
    kCluster = args["clusters"]
    
    inicio = time.time()
    
    # O código lê a imagem, converte-a para o espaço de cores LAB e a remodela para o clustering.
    # O código cria um histograma de cores para visualização.
    image = cv2.imread(filename)
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    imageOrig = image
    image = image.reshape((image.shape[0] * image.shape[1], 3))


    # Em seguida, usa o MiniBatchKMeans para gerar as etiquetas de cluster e os centros de cluster,
    # e aplica as etiquetas à imagem para criar a versão quantizada.
    clt = MiniBatchKMeans(n_init=3, n_clusters = kCluster, max_iter= 500)
    labels = clt.fit_predict(image)
    
    #MAGIC
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    cv2.imwrite(f'images/quantization/{filename.split("/")[-1].replace(" ","")}', quant) 


    image = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))


    # O código então usa o KMeans para reclusterizar a imagem quantizada para encontrar as cores dominantes,
    # e gera um histograma de cores e uma barra de cores para visualização.
    clt = KMeans(n_init=3, n_clusters = kCluster)
    clt.fit(image)
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    cv2.imwrite(f'images/kmeans/{filename.split("/")[1]}', cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))


    fim = time.time()
    print("Tempo de Execucao: {a:.3f} segs\n".format(a=(fim - inicio)))
    
    # Finalmente, ele exibe a imagem original, a imagem quantizada e a barra de cores.
    show_images([cv2.cvtColor(imageOrig, cv2.COLOR_LAB2RGB),cv2.cvtColor(quant, cv2.COLOR_BGR2RGB),bar], 1, ['Original', 'Quantizada', 'Quantidade'])

if __name__=='__main__':
    mainMenu()
