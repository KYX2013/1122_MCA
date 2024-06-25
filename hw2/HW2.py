import cv2
import os
import numpy as np

frame_features = []
frame_similarity = []
diff = []

def features_extraction(image_gray):
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    frame_features.append(hist)

boundary = []

def shot_change_boundary():
    mu, vari = np.mean(diff), np.var(diff)
    tb, ts = mu+3.5*vari, mu+0.2*vari
    period = 8
    gradual = []
    cumulate=0
    for x in range(len(frame_similarity)):
        s = frame_similarity[x]
        if s>tb:
            boundary.append(x+1)
            gradual = []
            continue
        elif s>ts:
            gradual.append(x+1)
            cumulate += (s-ts)
        else:
            if cumulate+ts>=tb and len(gradual)<period:
                boundary.extend(gradual)
                gradual = []
                cumulate = 0
            else:
                gradual = []
                cumulate = 0

        if len(gradual)>=period:
            if cumulate+ts>=tb:
                boundary.extend(gradual)
            gradual = []
            cumulate=0



def similarity_cal(frm1,frm2):
    score=np.sum(np.abs(frm1-frm2))
    frame_similarity.append(score)
    diff.append(np.mean(np.abs(frm1-frm2)))

def write_boundaries(file_path):
    flag = False
    with open(file_path, 'w') as file:
        file.truncate(0)
        for i in range(len(boundary)):
            if i == 0:
                file.write(str(boundary[i]))
            elif boundary[i] == boundary[i - 1] + 1:
                if flag==False:
                    flag = True
                    file.write(' ~ ')
                continue
            else:
                if flag:
                    file.write(f"{boundary[i - 1]}")
                    flag = False
                file.write(f"\n{boundary[i]}")
        if flag:
            file.write(f"{boundary[len(boundary)-1]}")
        if len(boundary) > 1 and boundary[-1] != boundary[-2] + 1:
            file.write(f" ~ {boundary[-1]}")


if __name__ == "__main__":
    dirPath = 'c:/Users/e9407/Documents/_Course/112_2/MCA/hw2/'
    video = ['climate','news','ngc']
    for v in video:
        files = os.listdir(dirPath+v+'_out')
        for fn in files:
            img = cv2.imread(dirPath+v+'_out/'+fn,cv2.IMREAD_GRAYSCALE)
            features_extraction(img)
        
        for i in range(1,len(frame_features)):
            similarity_cal(frame_features[i-1],frame_features[i])
        
        shot_change_boundary()
        write_boundaries(dirPath+v+'_detect.txt')

        print('Shot Change Detection of video '+v+':\n',boundary)
        boundary = []
        frame_similarity = []
        frame_features = []
