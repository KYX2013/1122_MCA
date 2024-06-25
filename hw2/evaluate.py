import os

def read_boundaries(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        boundaries = []
        for line in lines:
            if '~' in line:
                start, end = map(int, line.strip().split('~'))
                boundaries.extend(range(start, end + 1))
            else:
                try:
                    boundary = int(line.strip())
                    boundaries.append(boundary)
                except ValueError:
                    continue
    return boundaries

def compute_metrics(detected_boundaries, ground_truth_boundaries, total_frames):
    tp = len(set(detected_boundaries) & set(ground_truth_boundaries))
    fp = len(detected_boundaries) - tp
    fn = len(ground_truth_boundaries) - tp
    tn = total_frames - tp - fp - fn
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    matrix = [[tp,fp],
              [fn,tn]]
    
    return precision, recall, matrix


# Example usage
dirPath = 'c:/Users/e9407/Documents/_Course/112_2/MCA/hw2/'
video = ['climate','news','ngc']
for v in video:
    files = os.listdir(dirPath+v+'_out')
    detected_boundaries = read_boundaries(dirPath+v+'_detect.txt')
    ground_truth_boundaries = read_boundaries(dirPath+v+'_ground.txt')
    total_frames = len(files)

    precision, recall, mat = compute_metrics(detected_boundaries, ground_truth_boundaries, total_frames)
    print('\nEvaluation Result Of '+v+' :')
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", mat[0], '\n', mat[1])

