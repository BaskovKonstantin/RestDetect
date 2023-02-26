# На данны момент рабтате не стабильно и только по одному конкретному заведению
import cv2
import argparse
from openvino.inference_engine import IECore



ie = IECore()  #создается объект класса IECore

confidence = 0.2
error = 0.0
lag = 3
approveNum = 120
person_detect_patch = ''

buff_res = [
    [],
    [],
    []
]

# Инициализация предъобученной модели
def model_init():

    net_PVB = ie.read_network('intel/person-detection-0303/FP32/person-detection-0303.xml', 'intel/person-detection-0303/FP32/person-detection-0303.bin')
    exec_net_PVB = ie.load_network(net_PVB, 'CPU')

    return net_PVB, exec_net_PVB


# Обнаружение пересоны на заданном изображение
def person_detection (frame, net_PVB,  exec_net_PVB):

    dim = (1280, 720)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    inp = [resized.transpose(2, 0, 1)]
    input_name = next(iter(net_PVB.input_info))
    outputs = exec_net_PVB.infer({input_name: inp})
    outs = next(iter(outputs.values()))
    filtred_outs = []
    for i in outs:
        if (i[4] == 0):
            break
        elif( i[4] > confidence):
            filtred_outs.append(i)
    return filtred_outs, resized

# Отрисовка прямоугольников сингнализирующих о занятом/свободном месте
def drawDetected(frame, outs):
    for out in outs:
        coords = []
        if (out[4] == 0):
            break
        elif( out[4] > confidence):

            x_min = int(out[0])
            y_min = int(out[1])
            x_max = int(out[2])
            y_max = int(out[3])

            coords.append([x_min,y_min,x_max,y_max])
            coord = coords[0]

            # coord = coord* np.array([w, h, w, h])
            # coord = coord.astype(np.int32)

            print('confidence ',out[4])
            # cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(255, 0, 0), thickness = 2)
    return frame


# Отрисовка карты
def drawTabeArea(frame, tableList,  outs ):
    results = [0]*len(tableList)
    colors = [(0,0,0)]*len(tableList)

    for i in range(len(tableList)):
        color  = (128, 128, 128)

        res = 0

        for out in outs:

            if (out[4] == 0):
                break

            x_min = int(out[0])
            y_min = int(out[1])
            x_max = int(out[2])
            y_max = int(out[3])

            if ((tableList[i][0][0] < x_min * (1 + error) and  tableList[i][0][1] < y_min * (1 + error))
                    and
                (tableList[i][1][0] > x_max * (1 - error) and tableList[i][1][1] > y_max * (1 - error))):

                # print('GREEN')
                res = 1
                coef = sum(buff_res[i][-approveNum:])/approveNum
                # print(coef)
                if (coef > 0.5):
                    color=(0, int(255 * coef), 0)
                else:
                    color = (0, 0, int(255 * (1 - coef)))
                break
            else:
                # print('RED')
                res = 0
                coef = (approveNum - sum(buff_res[i][-approveNum:]))/approveNum
                # print(coef)
                if (coef > 0.5):
                    color = (0, 0, int(255 * coef))
                else:
                    color = (0, int(255 * (1 - coef)), 0)

        buff_res[i].append(res)
        colors[i] = color
        results[i] = res


        cv2.rectangle(frame, tableList[i][0], tableList[i][1], color=color, thickness=3)
    for i in range(len(buff_res)):
        print(i,' ',buff_res[i][-20:])
    return frame, results, colors


def drawMap(results, colors):

    img = cv2.imread("Canvas.png")

    h = img.shape[0]
    w = img.shape[1]

    square = 140

    mapList = [
        ((int(w * 0.2), int(h * 0.2)), (int(w * 0.2 + square), int(h * 0.2 + square))),
        ((int(w * 0.4), int(h * 0.2)), (int(w * 0.4 + square), int(h * 0.2 + square))),
        ((int(w * 0.3), int(h * 0.6)), (int(w * 0.3 + square), int(h * 0.6 + square)))
    ]

    border_color = (128, 128, 128)
    window_color = (255, 0, 0)

    cv2.rectangle(img, (int(w * 0.1), int(h * 0.1)), (int(w * 0.8), int(h * 0.95)), color=border_color, thickness=4)
    cv2.rectangle(img, (int(w * 0.6), int(h * 0.2)), (int(w * 0.7), int(h * 0.9)), color=border_color, thickness=-1)
    cv2.line(img, (int(w * 0.1), int(h * 0.2)), (int(w * 0.1), int(h * 0.5)), window_color, thickness = 4)
    cv2.line(img, (int(w * 0.1), int(h * 0.65)), (int(w * 0.1), int(h * 0.9)), window_color, thickness=4)

    for i in range(len(mapList)):
        color = (0, 255, 255)
        # print(results)
        if (results[i] == 1):
                coef = (sum(buff_res[i][-approveNum:]))/approveNum
                print(coef)
                color = colors[i]
                # print('GREEN')
        else:
                # print('RED')
                coef = (approveNum - sum(buff_res[i][-approveNum:]))/approveNum
                print(coef)
                color = colors[i]


        # print(img.shape)
        # print(mapList)
        cv2.rectangle(img, mapList[i][0], mapList[i][1], color=color, thickness=-1)
    return img

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Имя видефайла для захвата', default='video4.mp4')
    args = parser.parse_args()
    video_file_name = "video\\" + args.filename

    net_PVB, exec_net_PVB = model_init()


    # Инициализируем объект захвата видео
    cap = cv2.VideoCapture(video_file_name)
    # Захватываем кадр из видео
    ret, frame = cap.read()


    #вызов функции,в которой происходит обнаружение людей /транспортных средств /велосипедов
    while ret:

        outs ,frame = person_detection (frame, net_PVB, exec_net_PVB)
        frame = drawDetected(frame, outs)

        h = frame.shape[0]
        w = frame.shape[1]

        tableList = [
            ((int(w*0.15), int(h*0.35) ),(int(w*0.3),int(h*0.65) )),
            ((int(w*0.32), int(h*0.2) ),(int(w*0.5),int(h*0.55) )),
            ((int(w*0.15), int(h*0.575) ),(int(w*0.45),int(h*0.9) ))

        ]
        frame, results, colors = drawTabeArea(frame, tableList, outs)


        img = drawMap( results, colors)

        cv2.imshow('map', img)
        cv2.waitKey(1)

        cv2.imshow('Frame.jpg', frame) #вывод на экран изображения, на котором выделены люди / транспортных средства / велосипеды
        cv2.waitKey(1)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()