from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
import dlib
from imutils import face_utils

def facecrop(org_path, num_frames, face_detector, face_predictor):
    min_frame_count = np.float32('inf')

    for tmp_path in org_path:
        cap_org = cv2.VideoCapture(tmp_path)
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count_org < min_frame_count:
            min_frame_count = frame_count_org

    frame_idxs = np.linspace(0, min_frame_count - 1, num_frames, dtype=int)

    for tmp_path in org_path:
        cap_org = cv2.VideoCapture(tmp_path)
        for cnt_frame in range(min_frame_count):
            ret_org, frame_org = cap_org.read()
            if not ret_org:
                tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, tmp_path))
                break
            if cnt_frame not in frame_idxs:
                continue

            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
            faces = face_detector(frame, 1)
            if len(faces) == 0:
                tqdm.write('Frame read {} Error! do not find the face: {}'.format(cnt_frame, tmp_path))
                continue

            landmarks = []
            size_list = []
            for face_idx in range(len(faces)):
                landmark = face_predictor(frame, faces[face_idx])
                landmark = face_utils.shape_to_np(landmark)
                x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
                x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
                face_s = (x1 - x0) * (y1 - y0)
                size_list.append(face_s)
                landmarks.append(landmark)
            landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
            landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

            save_path_ = os.path.dirname(tmp_path).replace('videos', 'frames/') + os.path.basename(tmp_path).replace('.mp4','/')
            os.makedirs(save_path_, exist_ok=True)
            image_path = save_path_ + str(cnt_frame).zfill(3) + '.png'
            land_path = save_path_ + str(cnt_frame).zfill(3)

            land_path = land_path.replace('/frames', '/landmarks')

            os.makedirs(os.path.dirname(land_path), exist_ok=True)
            np.save(land_path, landmarks)

            if not os.path.isfile(image_path):
                cv2.imwrite(image_path, frame_org)
        cap_org.release()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset',
                        choices=['DeepFakeDetection_original', 'DeepFakeDetection', 'FaceShifter', 'Face2Face',
                                 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Original', 'Celeb-real', 'Celeb-synthesis',
                                 'YouTube-real', 'DFDC', 'DFDCP', 'ALL'], default='Deepfakes')

    parser.add_argument('-p', dest='dataset_path', default="/media/lhz/Data/FF++")
    parser.add_argument('-c', dest='comp', choices=['raw', 'c23', 'c23'], default='c23')
    parser.add_argument('-n', dest='num_frames', type=int, default=10)
    args = parser.parse_args()
    dataset_path = []
    if args.dataset == 'Original':
        dataset_path.append(os.path.join(args.dataset_path, 'original_sequences/youtube/{}/videos/'.format(args.comp)))
    elif args.dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path.append(os.path.join(args.dataset_path, 'manipulated_sequences/{}/{}/videos/'.format(args.dataset, args.comp)))
    elif args.dataset == 'ALL':
        dataset_path.append(os.path.join(args.dataset_path, 'original_sequences/youtube/{}/videos/'.format(args.comp)))
        dataset_path.append(os.path.join(args.dataset_path, 'manipulated_sequences/{}/{}/videos/'.format('FaceShifter', args.comp)))
        dataset_path.append(os.path.join(args.dataset_path, 'manipulated_sequences/{}/{}/videos/'.format('Face2Face', args.comp)))
        dataset_path.append(os.path.join(args.dataset_path, 'manipulated_sequences/{}/{}/videos/'.format('Deepfakes', args.comp)))
        dataset_path.append(os.path.join(args.dataset_path, 'manipulated_sequences/{}/{}/videos/'.format('FaceSwap', args.comp)))
        dataset_path.append(os.path.join(args.dataset_path, 'manipulated_sequences/{}/{}/videos/'.format('NeuralTextures', args.comp)))
    else:
        raise NotImplementedError

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    video_list = []
    for tmp_path in dataset_path:
        video = sorted(glob(tmp_path+'*'))
        video_list.append(video)
    video_list = np.array(video_list)
    for i in tqdm(range(1000)):
        facecrop(video_list[:, i], num_frames=args.num_frames, face_predictor=face_predictor,
                 face_detector=face_detector)
