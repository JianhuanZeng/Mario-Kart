import sys, logging, os, argparse
import numpy as np
from PIL import Image, ImageGrab
from socketserver import TCPServer, StreamRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# deep model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv3D,Conv2D,BatchNormalization

OUT_SHAPE = 1
IN_DEPTH = 4
INPUT_WIDTH = 200
INPUT_HEIGHT = 100
INPUT_CHANNELS = 3


def sequential_model():
    model = Sequential()
    keep_prob = 0.8

    # Sequential NVIDIA's model
    model.add(BatchNormalization(input_shape=(IN_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))
    model.add(Conv3D(24, kernel_size=(2, 5, 5), strides=(2, 2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(36, kernel_size=(2, 5, 5), strides=(1, 2, 2), activation='relu'))
    model.add(Reshape(model.output_shape[2:], input_shape=model.output_shape[1:]))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign', name="predictions"))
    return model

def prepare_image(im, im_prd):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_prd[0, -1, :, :, :] = im_arr
    return im_prd


class TCPHandler(StreamRequestHandler):
    def handle(self):
        # initialized
        try:
            previous_steers
        except:
            print('initialization image prediction ...')
            im_prd = np.zeros((1, IN_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
            previous_steers = np.zeros(3)
            attention = [0.027, 0.09, 0.3]


        logger.info("Handling a new connection...")
        for line in self.rfile:
            message = str(line.strip(), 'utf-8')
            logger.debug(message)

            if message.startswith("COURSE:") and not args.all:
                course = message[7:].strip().lower()
                weights_file = 'weights/{}.hdf5'.format(course)
                logger.info("Loading {}...".format(weights_file))
                model.load_weights(weights_file)

            if message.startswith("PREDICTFROMCLIPBOARD"):
                im = ImageGrab.grabclipboard()
                if im != None:
                    prediction_steer = model.predict(prepare_image(im, im_prd), batch_size=1)[0]
                    prediction_steer = np.sum(previous_steers*np.array(attention))+prediction_steer[0]
                    prediction_steer = max(min(prediction_steer, 1),-1)
                    self.wfile.write((str(prediction_steer) + "\n").encode('utf-8'))
                else:
                    self.wfile.write("PREDICTIONERROR\n".encode('utf-8'))

            if message.startswith("PREDICT:"):
                im = Image.open(message[8:])
                # update input frames
                im_prd = prepare_image(im, im_prd)
                prediction_steer = model.predict(im_prd, batch_size=1)[0]
                prediction_steer = np.sum(previous_steers * np.array(attention)) + prediction_steer[0]
                prediction_steer = max(min(prediction_steer, 1), -1)
                # update input steerings
                previous_steers = np.append(previous_steers[1:], prediction_steer)
                # send to emulator
                self.wfile.write((str(prediction_steer) + "\n").encode('utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a prediction server that other apps will call into.')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Use the combined weights for all tracks, rather than selecting the weights file based off of the course code sent by the Play.lua script.',
                        default=False)
    parser.add_argument('-p', '--port', type=int, help='Port number', default=36296)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading model...")
    model = sequential_model(keep_prob=1)


    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', args.port), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
