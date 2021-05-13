import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import skimage
import subprocess
import csv
import time
import IPython
import sys
from pathlib import Path
from skimage import transform, color, feature
from packaging import version
from io import StringIO

import warnings
warnings.filterwarnings("ignore", message="Persisting input arguments took")


# --- DATASET ---

def shuffle_in_unison(elements, seed):
    prev_seed = np.random.get_state()
    
    np.random.seed(seed)
    rng_state = np.random.get_state()
    for x in elements:
        np.random.set_state(rng_state)
        np.random.shuffle(x)
    
    np.random.set_state(prev_seed)

def load_labeled_dataset_from_txt(filePath, featureCount):
    data = np.loadtxt(filePath, usecols=range(0, featureCount))
    labels = np.loadtxt(filePath, usecols=[featureCount])

    return data, labels

def internal_load_labeled_dataset(filelist_name, db_folder_path):
    files = []
    labels = []
    
    db_folder_path = Path(db_folder_path)
    
    with open(str(db_folder_path / filelist_name), 'r') as f:
        for i, line in enumerate(f):
            path, label = line.split()
            files.append(db_folder_path / path)
            labels.append(int(label))

    images = [plt.imread(file_path) for file_path in files]

    return images, np.array(labels)

def load_labeled_dataset(filelist_name, db_folder_path, cache=None):
    if cache is not None:
        cached_function = cache.cache(internal_load_labeled_dataset)
    else:
        cached_function = internal_load_labeled_dataset

    return cached_function(filelist_name, db_folder_path)
	
def load_label_names(filelabel_name, db_folder_path):
    with open(str(Path(db_folder_path) / filelabel_name)) as f:
        content = f.readlines()
            
    return np.array([x.strip() for x in content])
	
def separate_pattern_classes(data, labels, class_count):
    patternCount=data.shape[0]
    result = np.zeros(class_count, dtype = np.ndarray)
    for i in range(class_count):
        result[i] = np.array([data[j] for j in range(patternCount) if labels[j]==i])

    return result

# ---
	
# --- IMAGES ---
	
def internal_resize_images(imgs, size_h, size_w):
    # Si assume che l'immagini di input siano RGB (3-canali) di dimensioni (h, w, c)
    num_imgs = len(imgs)
    resized_images = np.empty((num_imgs, size_h, size_w, 3), dtype=np.float32)
    
    for i in range(num_imgs):
        if version.parse(skimage.__version__) < version.parse("0.14.0"):
            res_img = transform.resize(imgs[i], (size_h, size_w), mode='reflect')
        else:
            res_img = transform.resize(imgs[i], (size_h, size_w), mode='reflect', anti_aliasing=True)
        resized_images[i] = res_img
        
    return resized_images

def resize_images(imgs, size_h, size_w, cache=None):
    if cache is not None:
        cached_function = cache.cache(internal_resize_images)
    else:
        cached_function = internal_resize_images

    return cached_function(imgs, size_h, size_w)

# ---

# --- FEATURES ---

def internal_extract_hog(images,convert_to_gray,orientations,pixels_per_cell,cells_per_block):
    # Lorenzo: "visualise" is deprecated in recent versions of skimage (in favor of "visualize")!
    hog_features = []
    for idx, img in enumerate(images):
        image=img
        if convert_to_gray:
            image=color.rgb2gray(image)
        hog = feature.hog(image,orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,block_norm='L2-Hys',visualise=False) 
        hog_features.append(hog)
        
    return hog_features

def extract_hog(images,convert_to_gray,orientations,pixels_per_cell,cells_per_block,cache=None):
    if cache is not None:
        cached_function = cache.cache(internal_extract_hog)
    else:
        cached_function = internal_extract_hog

    return cached_function(images,convert_to_gray,orientations,pixels_per_cell,cells_per_block)

def compute_square_euclidean_distance(v1, v2):
    return sum(np.power(v1 - v2, 2))
	
	
# ---

# --- GPU ---

LOCK_FILE_PATH = '/home/esercitazioni_ml18/gpu_lock%d.lock'
MAX_USERS_PER_GPU = 3
DEFAULT_GPU_PATH = Path.cwd() / 'default_gpu.txt'


def run_nvidia_smi(*args):
    result = subprocess.run(['nvidia-smi', '--format=csv,noheader,nounits', *args],
                            stdout=subprocess.PIPE).stdout.decode('utf-8')
    f = StringIO(result)
    reader = csv.reader(f)
    rows = []
    for row in reader:
        rows += [[field.strip() for field in row]]
    return rows

def get_gpus():
    result = run_nvidia_smi('--query-gpu=index,name,memory.total')
    gpus = [None for _ in range(len(result))]
    for gpu in result:
        gpus[int(gpu[0])] = {'name': gpu[1], 'memory.total': float(gpu[2])}
    return gpus

def get_gpu_processes(index):
    result = run_nvidia_smi('--query-compute-apps=pid,used_memory', '--id=%d' % index)
    processes = {}
    for process in result:
        processes[process[0]] = float(process[1])
    return processes

def process_count_per_gpu():
    gpus = get_gpus()
    return [len(get_gpu_processes(i)) for i in range(len(gpus))]

def ram_state_per_gpu():
    gpus = get_gpus()
    print(get_gpus())

    usage_per_gpu = np.empty((len(gpus),))
    max_per_gpu =  np.empty((len(gpus),))
    for i in range(len(gpus)):
        processes = get_gpu_processes(i)
        usages = np.array([processes[k] for k in processes])
        usage_per_gpu[i] = usages.sum()
        max_per_gpu[i] = gpus[i]['memory.total']

    return usage_per_gpu, max_per_gpu

def acquire_lock(lock_path):
    from filelock import FileLock
    lock_path = Path(lock_path)

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
    except:
        pass

    lock_object = FileLock(lock_path)

    with lock_object:
        time.sleep(3)
        return lock_object

def acquire_gpu(gpu_id=-1,
                lock_path=LOCK_FILE_PATH,
                max_per_gpu=MAX_USERS_PER_GPU,
                default_gpu_path=DEFAULT_GPU_PATH,
                cpu_only=False):
    if cpu_only:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return -1
    
    gpu_id = int(gpu_id)
    if gpu_id < 0:
        default_gpu_path = Path(default_gpu_path)
        if default_gpu_path.exists():
            with open(default_gpu_path, 'r') as f:
                line = f.readline()
                if line == '':
                    raise ValueError('Error allocating GPU: no GPU specified')
                gpu_id = int(line)
        else:
            print('Error allocating GPU: no GPU specified')
            raise ValueError('Error allocating GPU: no GPU specified')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print('Allocating GPU %d... ' % gpu_id, end="")
    gpus = get_gpus()

    if gpu_id >= len(gpus):
        print('Bad GPU id, max index is', len(gpus))
        raise ValueError('Error allocating GPU: no GPU specified')

    with open(default_gpu_path, 'w') as f:
        f.write(str(gpu_id))

    lock_path = lock_path % gpu_id
    lock = acquire_lock(lock_path=lock_path)
    result = -1
    while result == -1:
        with lock:
            my_pid = os.getpid()
            processes = get_gpu_processes(gpu_id)
            if len(processes) < max_per_gpu or str(my_pid) in processes:
                result = gpu_id
                import tensorflow as tf
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.25
                with tf.Session(config=config) as sess:
                    pass
    
    print('Done! (%d: %s)' % (result, gpus[result]['name']))
    return result

def release_gpu():
    print('GPU freed!')
    return IPython.core.display.HTML("<script>Jupyter.notebook.kernel.restart()</script>")

# ---

# --- TENSORFLOW ---

def tf_create_graph(class_num, tf,optimizer='momentum'):
    import tensorflow.contrib.slim as slim
    import mobilenet_v1
    
    image_side = 128
    depth_multiplier=0.5
    # File dei pesi del modello pretrained su ImageNet (deve corrispondere ai moltiplicatori sopra)
    ImageNet_weights = '/home/esercitazioni_ml18/DBs/Models/mobilenet_v1_0.5_128/mobilenet_v1_0.5_128.ckpt'
  
    height = width = image_side

    tf.reset_default_graph()
    tf.set_random_seed(1234)

    print("Creating MobileNet Graph ...")
    # placeholder per pattern ed etichette
    X = tf.placeholder(tf.float32, shape=[None, height, width, 3], name="X")
    y = tf.placeholder(tf.int32, shape=[None])
    # placeholder booleano, che servirà per indicare alla rete se sta operando in training o in inference
    training = tf.placeholder_with_default(False, shape=[])

    # Creazione (tramite slim) del modello di MobileNet desiderato
    with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        logits, end_points = mobilenet_v1.mobilenet_v1(X, depth_multiplier = depth_multiplier, num_classes=1001, is_training=training)
    original_net_saver = tf.train.Saver()   # necessario durante la sessione per caricare pesi dal modello originale pretrained

    # Punto di aggancio (prima del logit 1001 classi) -> Shape (?,1,1,1024) per il nuovo livello di output
    anchor = end_points['AvgPool_1a']   # end_point è un dictionary che ci permette di accedere per nome ai livelli del modello
    prelogits = tf.squeeze(anchor, axis=[1, 2])  # -> Aggiustiamo la shape a (?,1024)
    # Aggiungiamo nuovo livello di output con il numero di classi desiderate 
    new_logits = tf.layers.dense(prelogits, class_num, name="new_logits")
    # Aggiungiamo anche softmax (non obbligatorio visto che nella loss è già integrato softmax, ma utile per ottenere le predictions come probabilità)
    Y_proba = tf.nn.softmax(new_logits, name="Y_proba")

    # loss function e ottimizzazione
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new_logits, labels=y)
    loss = tf.reduce_mean(cross_entropy)
    if optimizer=='momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum = 0.9)
    elif optimizer=='adam':
        optimizer = tf.train.AdamOptimizer()
    else:
        raise ValueError('Optimizer non supportato. Per utilizzarlo è necessario aggiornare manualmente il file utils.py.')
    training_op = optimizer.minimize(loss)

    # Valutazione accuratezza
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # inizializzatore variabili
    init = tf.global_variables_initializer()

    # saver del grafo modificato (con nuovo livelli di output)
    saver = tf.train.Saver() 

    print("Graph created")
    return saver, accuracy, X, y, training, Y_proba
	
def tf_compute_test_accuracy(test_x, test_y, minibatch_size, class_num, sess, accuracy, X, y, training, Y_proba):
    n_iterations_per_epoch_test = test_x.shape[0] // minibatch_size
    test_accuracy = 0
    probs = np.empty((test_x.shape[0], class_num))
    for iteration in range(n_iterations_per_epoch_test):
        print("+", end="", flush = True)
        # get current minibatches    
        start = iteration * minibatch_size
        end = (iteration + 1) * minibatch_size
        X_minibatch = test_x[start:end]
        y_minibatch = test_y[start:end]
        minibatch_acc, minibatch_prob = sess.run([accuracy, Y_proba], feed_dict={X: X_minibatch, y: y_minibatch, training: False})
        test_accuracy += minibatch_acc
        probs[start:end] = minibatch_prob
    return test_accuracy / n_iterations_per_epoch_test, probs
	
# ---

