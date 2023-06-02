# Machine learning for analyzing egocentric videos of patients during rehabilitation

Please, make sure to follow all those steps to run this code project and get the same results as I did.

If you don't want to do the hands and object detection part, please skip the (optional) steps and start directly at the "Final dataset creation" step, using the data saved in the pickle_data folder and the "object_detected.csv" file.

### Get the data
Before doing anything, you need to make sure that all the data is in your possession. All the egocentric video frames used in this project are available on the nyx server of HEIG-VD, in the "Sam_Salad_Preparation" folder, in the shared_project section.

### Hands detection (optional)
Run the "hands_detection_MediaPipe_center.ipynb" notebook two times : once with the frames contained in the 300_300_AfC folder, available on the nyx server (300x300 pixels frames obtained by cropping method) and second time with the 300_300_AfP folder (300x300 pixels frames obtained by padding method).

Run the "hands_detection_MediaPipe_all_landmarks.ipynb" notebook two times : once with the 300_300_AfC folder and second time with the 300_300_AfP folder.

### Merging hands detection (optional)
Once you have run both of the previous notebook twice, you should have generate the following pickle files :

  - 300_300_AfC_hands_coordinates.pickle
  - 300_300_AfP_hands_coordinates.pickle
  - 300_300_AfC_hands_coordinates_21_landmarks.pickle
  - 300_300_AfP_hands_coordinates_21_landmarks.pickle

Run the "merging_hands_detection_center.ipynb" and the "merging_hands_detection_21_landmarks.ipynb" notebooks to merge those results and to get the following files :

  - 300_300_AfC_AfP_hands_coordinates_center.pickle
  - 300_300_AfC_AfP_hands_coordinates_21_landmarks.pickle

### Object detection (optional)
To detect object on the frames like I did, you need to use YOLOv7. (If you need more details, everything is clearly explained in the report of the project :D)

To do so, please follow those steps (in a Linux environnement) :
  - apt-get install wget
  - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sha256sum Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
  - conda create -n yolov7 python=3.8
    conda activate yolov7
  - !git clone https://github.com/WongKinYiu/yolov7
  - cd yolov7
    pip install -r requirements.txt
  - python detect.py --weights yolov7-e6e.pt --conf 0.80 --img-size 640 --source  --save-txt --save-conf --nosave

With those steps, you should be able to perform the object detection.

Run "object_detection_YOLOv7.ipynb" notebook to clean the objects detection made by YOLOv7.

### Final dataset creation
Run "dataset_creation_no_images.ipynb" to create the final dataset that will be used to train classification models.

### Models creation and training
Run the "submovement_classification_models.ipynb" notebook to perform the 10 classes classification.
Run the "submovement_binary_classification_model.ipynb" notebook, to perform the binary classification.
