import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import argparse
import mvnc.mvncapi as mvnc

AGE_GRAPH = './graph/AgeNet.graph'
GENDER_GRAPH = './graph/GenderNet.graph'
INPUT_DIMENSION = (227, 227)

AGE_LIST = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-120']
GENDER_LIST = ['Male', 'Female']

def human_distance(enc1, enc2): 
    return np.sqrt(np.sum(np.square(normalize(enc1) - normalize(enc2))))

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.enumerate_devices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.open()

    return device

def load_graph(device, graph_path):

    # Read the graph file into a buffer
    with open( graph_path, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = mvnc.Graph(graph_path)
    # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

    return graph, fifo_in, fifo_out

def pre_process_image(image_path, mean_file):
    ilsvrc_mean = np.load(mean_file).mean(1).mean(1) #loading the mean file
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,INPUT_DIMENSION)
    img = img.astype(np.float32)
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])
    return img

def infer_image(graph, img, fifo_in, fifo_out):

    # The first inference takes an additional ~20ms due to memory 
    # initializations, so we make a 'dummy forward pass'.
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img.astype(np.float32), None)

    output, userobj = fifo_out.read_elem()

    # Load the image as an array
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img.astype(np.float32), None)

    # Get the results from NCS
    output, userobj = fifo_out.read_elem()


    # Get execution time
    inference_time = graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)

    # Print the results
    print("Execution time: " + str(np.sum( inference_time )) + "ms")
    output = np.expand_dims(output,axis=0)
    return output

def clean_up(device, graph, fifo_in, fifo_out):
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="path to image",
                    required=True)
    parser.add_argument("--mean_file", required=True)
    args = parser.parse_args()
    img_path = args.image_path
    mean_file = args.mean_file
    img = pre_process_image(img_path, mean_file)
    
    device = open_ncs_device()
    age_graph, age_fifo_in, age_fifo_out = load_graph(device, AGE_GRAPH)
    age_out = infer_image(age_graph, img, age_fifo_in, age_fifo_out)[0]
    age_pred = AGE_LIST[age_out.argsort()[-1]]
    age_prob = age_out[age_out.argsort()[-1]]
    print("Age predicted: %s, %.2f%%" % (age_pred, age_prob*100))
    clean_up(device, age_graph, age_fifo_in, age_fifo_out)
    
if __name__ == "__main__":
    sys.exit(main())
