import tkinter as tk
from PIL import ImageTk, Image
import csv, os
import pandas as pd
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

OUTPUTS_FILEPATH        = r"data\outputs2.tsv"
RESUME_INDEX_FILEPATH   = r"data\resume_index2.txt" 
BBOX_CSV_FILEPATH       = r"data\roboflow2\Khoi.csv"
IMG_FOLDER_PATH = r"data\roboflow2\bill.v1i.tensorflow\train"

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)

global all_bbox_info
global bbox_info
def add_index_col_to_csvfile(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if 'index' in df.columns: # no need to add
        print('Index col existed. No need to add.')
        return
    
    df['index'] = df.index
    # rearrange columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv(csv_file_path, index=False)
    print('Index col added to csv.')
def init_bbox(bbox_csv_filepath, resume_index_filepath):
    global all_bbox_info
    global bbox_info

    def get_resume_index(resume_index_filepath):
        try:
            with open(resume_index_filepath, 'r') as f:
                try:
                    current_index = int(f.read())
                except:
                    current_index=0
        except:
            current_index = 0
        return current_index
    def get_bbox_info_fromcsv(bbox_csv_filepath): 
        with open(bbox_csv_filepath, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for bbox in csv_reader:
                # print(bbox)
                yield bbox

    all_bbox_info = get_bbox_info_fromcsv(bbox_csv_filepath)
    # bbox_info = next(all_bbox_info)  # first row: title # filename,width,height,class,xmin,ymin,xmax,ymax
    # bbox_info = next(all_bbox_info)  # second row:  # mcocr_public_145014ikozz_jpg.rf.09efb3664ef5e74c8f51bc8531d1df46.jpg,440,403,TITLE,92,81,335,105
    current_index = get_resume_index(resume_index_filepath)
    for _ in range(current_index+2): 
        bbox_info = next(all_bbox_info)    
def get_cropped_img(bbox_info):
    def crop_resize(img:Image, latlong=None, max_img_size=(600, 80)):
        '''
            img: origin img
            latlong: latitude&longtitude
            max_img_size: maximum size to resize the image to
        '''
        # crop
        left, top, right, bottom = latlong
        edges_to_crop = (left, top, right, bottom)
        cropped_img = img.crop(edges_to_crop)
        # resize
        w, h = cropped_img.size
        ratio = min(max_img_size[0]/w, max_img_size[1]/h)
        newsize = (int(w*ratio), int(h*ratio))
        cropped_img = cropped_img.resize(newsize, Image.Resampling.LANCZOS)
        return cropped_img

    def get_latlong(bbox_info):
        left, top, right, bottom = [int(i) for i in bbox_info[5:9]]
        assert left < right and top < bottom
        latlong = (left, top, right, bottom)
        return latlong

    img_name = bbox_info[1]
    img_path = os.path.join(IMG_FOLDER_PATH, img_name)
    img = Image.open(img_path)
    cropped_img = crop_resize(img, get_latlong(bbox_info))
    return cropped_img
def tostring_bbox_info(bbox_info):
    return f'Index: {bbox_info[0]}\nLabel: {bbox_info[4]}' # \nImage name: {bbox_info[1]}
def next_bbox_button(): # command for the `Next` button
    def take_input_writetofile(outputs_filepath):
        global bbox_info
        # print('bbox info: ', bbox_info)
        content = entry_noidung.get()
        
        bbox_info.append(content)   
        entry_noidung.delete(0, tk.END)

        with open(outputs_filepath, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(bbox_info)
        status_str = 'Status : ' + str(bbox_info) + ' saved.'
        status_str = f'Status: Index {bbox_info[0]} annotated as [{content}]'
        label_status.config(text=status_str)
    
    take_input_writetofile(OUTPUTS_FILEPATH) # save current annotation before moving to the next one

    global bbox_info
    try:
        bbox_info = next(all_bbox_info)
    except StopIteration:
        print('All images have been annotated! This work is done.')
        label_status.config(text='Status: All images have been annotated! This work is done.')

    cropped_img = get_cropped_img(bbox_info)
    imgTK = ImageTk.PhotoImage(cropped_img)
    label_img.config(image=imgTK)
    label_img.image = imgTK
    entry_noidung.delete(0, tk.END)
    predicted_text = detector.predict(cropped_img)
    entry_noidung.insert(0, predicted_text)
    label_bbox.config(text=tostring_bbox_info(bbox_info))

    # save checkpoint
    current_index = int(bbox_info[0])
    with open(RESUME_INDEX_FILEPATH, 'w') as f:
        f.write(str(current_index))

add_index_col_to_csvfile(BBOX_CSV_FILEPATH)
init_bbox(BBOX_CSV_FILEPATH, RESUME_INDEX_FILEPATH)



window = tk.Tk()
window.title('IMPxDPL 2023')
window.geometry("1000x420")  
blankframe = tk.Frame(window, width=600, height=20)
blankframe.pack()


title_frame = tk.Frame(window, width=600, height=600)
title_frame.pack()

label_title = tk.Label(title_frame, text='Text Annotating', font=("Helvetica", 30))
label_title.pack(pady=(0, 50))

cropped_img = get_cropped_img(bbox_info)
imgTK = ImageTk.PhotoImage(cropped_img)
label_img =  tk.Label(title_frame, image=imgTK, height=80)
label_img.pack()

label_bbox = tk.Label(title_frame, text=tostring_bbox_info(bbox_info))
label_bbox.pack()

label_instruction = tk.Label(title_frame, text='Enter the text you see in the box below:')
label_instruction.pack(pady=(20, 0))

entry_noidung = tk.Entry(title_frame, width=60,font=('Georgia 12'))
predicted_text = detector.predict(cropped_img)
entry_noidung.insert(0, predicted_text)
entry_noidung.pack()

label_status = tk.Label(title_frame, text='Status: _')
label_status.pack(side='left')

frame_next_button = tk.Frame(window, width=600, height=200)
frame_next_button.pack()

button_next_item = tk.Button(frame_next_button, text='Next', command=next_bbox_button)
button_next_item.pack(side='bottom', pady=(50, 0))

window.mainloop()