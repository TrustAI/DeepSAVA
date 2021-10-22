"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
  
    data_file = []
    train_list = []
    test_list =[]
    val_list =[]
    folders = ['train', 'test']
    test_file = []
    for folder in folders:
        class_folders = sorted(glob.glob(folder + '/*'))
        clasn = 0
        for vid_class in class_folders:
            
            class_files = glob.glob(vid_class + '/*')
            ind = 0 
            for video_files in class_files:
                # Get the parts of the file.
                 
                video_parts = get_video_parts(video_files)

                train_or_test, classname, filename_no_ext = video_parts
                    

                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])
                if folder =='train':
                
                    train_list.append([filename_no_ext,video_files,nb_frames,clasn])
                if folder =='test':
                    
                    test_list.append([filename_no_ext,video_files,nb_frames,clasn])
                    test_file.append([train_or_test, classname, filename_no_ext, nb_frames])
                    
                print("Generated %d frames for " )
                ind +=1
            clasn +=1

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)
    with open('test_file.csv', 'w') as tout:
        writer = csv.writer(tout)
        writer.writerows(test_file)
    with open('trainlist.txt', 'w') as f:
        for item in train_list:
            line = ("{} {} {} {}\n".format(item[0], item[1],item[2],item[3]))
            
            f.write(line)
    with open('testlist.txt', 'w') as f2:
        for item in test_list:
            line = ("{} {} {} {}\n".format(item[0], item[1],item[2],item[3]))
            f2.write(line)
   

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext = video_parts
    generated_files = glob.glob(train_or_test + '/' + classname + '/' +
                                filename_no_ext + '/'+'*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[2]
    
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(train_or_test + '/' + classname +
                               '/' + filename_no_ext +'/'+ 'frame0001.jpg'))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
