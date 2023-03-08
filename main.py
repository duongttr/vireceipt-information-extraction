import tkinter as tk
from PIL import ImageTk, Image
import csv, os
import pandas as pd
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'

detector = Predictor(config)


output_bbox_and_contents_filepath = r'data/outputs.tsv'
bbox_csv_filepath = r"data\roboflow\train\_annotations.csv"
resume_index_filepath = r"data\resume_index.txt" # 

def add_index_col_to_csvfile(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if 'index' in df.columns: # no need to add
        print('Index col existed.')
        return
    
    # new column named 'index'
    df['index'] = df.index

    # # rearrange the columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv(csv_file_path, index=False)
    print('Index col added to csv.')

add_index_col_to_csvfile(bbox_csv_filepath)

def get_bbox_info_fromcsv(bbox_csv_filepath): 
    with open(bbox_csv_filepath, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for bbox in csv_reader:
            print(bbox)
            yield bbox

global all_bbox_info
global bbox_info
all_bbox_info = get_bbox_info_fromcsv(bbox_csv_filepath)
bbox_info = next(all_bbox_info)  # first row: title # filename,width,height,class,xmin,ymin,xmax,ymax
bbox_info = next(all_bbox_info)  # second row:  # mcocr_public_145014ikozz_jpg.rf.09efb3664ef5e74c8f51bc8531d1df46.jpg,440,403,TITLE,92,81,335,105

with open(resume_index_filepath, 'r') as f:
    try:
        current_index = int(f.read())
    except:
        current_index=0

for i in range(current_index):
    bbox_info = next(all_bbox_info)


global img_name
global img_path
img_name = bbox_info[1]
img_path = os.path.join('data', 'roboflow', 'train', img_name)

def crop_resize(img:Image, latlong=None, max_img_size=(600, 80)):
    '''
        img: origin img
        latlong: latitude&longtitude
        max_img_size: maximum size to resize the image to
    '''

    left, top, right, bottom = latlong
    edges_to_crop = (left, top, right, bottom)
    cropped_img = img.crop(edges_to_crop)
    # resize
    w, h = cropped_img.size

    ratio = min(max_img_size[0]/w, max_img_size[1]/h)
    newsize = (int(w*ratio), int(h*ratio))
    cropped_img = cropped_img.resize(newsize, Image.Resampling.LANCZOS)

    return cropped_img

def take_input_writetofile():
    global bbox_info
    print('bbox info: ', bbox_info)
    content = entry_noidung.get()
    
    bbox_info.append(content)   
    entry_noidung.delete(0, tk.END)

    with open(output_bbox_and_contents_filepath, 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(bbox_info)
    status_str = 'Status : ' + str(bbox_info) + ' saved.'
    status_str = f'Status: Index {bbox_info[0]} annotated as [{content}]'
    label_status.config(text=status_str)

def get_latlong(bbox_info):
    left, top, right, bottom = [int(i) for i in bbox_info[5:9]]
    assert left < right and top < bottom
    latlong = (left, top, right, bottom)
    return latlong

window = tk.Tk()
window.title('IMPxDPL 2023')
window.geometry("1000x420")  
blankframe = tk.Frame(window, width=600, height=20)
blankframe.pack()
title_frame = tk.Frame(window, width=600, height=600)
title_frame.pack()



label_title = tk.Label(title_frame, text='Text Annotating', font=("Helvetica", 30))
label_title.pack(pady=(0, 50))

label_img_name = tk.Label(title_frame, text=f'Img: {img_name}')
# label_img_name.pack()


img = Image.open(img_path)
cropped_img = crop_resize(img, get_latlong(bbox_info))
predicted_text = detector.predict(cropped_img)

imgTK = ImageTk.PhotoImage(cropped_img)

label_img =  tk.Label(title_frame, image=imgTK, height=80)
label_img.pack()

def display_bbox_info(bbox_info):
    return f'Index: {bbox_info[0]}\nLabel: {bbox_info[4]}' # \nImage name: {bbox_info[1]}

label_bbox = tk.Label(title_frame, text=display_bbox_info(bbox_info))
label_bbox.pack()

label_instruction = tk.Label(title_frame, text='Enter the text you see in the box below:')
label_instruction.pack(pady=(20, 0))


entry_noidung = tk.Entry(title_frame, width=100)
entry_noidung.insert(0, predicted_text)
entry_noidung.pack()

# button_Okay = tk.Button(title_frame, text='Okay&Save',command = take_input_writetofile)
# button_Okay.pack()

label_status = tk.Label(title_frame, text='Status: _')
label_status.pack(side='left')

def next_bbox():
    take_input_writetofile() # save current annotation before moving to the next one

    global bbox_info
    global img_name
    global img_path
    try:
        bbox_info = next(all_bbox_info)
    except StopIteration:
        print('All images have been annotated! This work is done.')
        label_status.config(text='Status: All images have been annotated! This work is done.')

    img_name = bbox_info[1]
    img_path = os.path.join('data', 'roboflow', 'train', img_name)
    print('imgpath: ', img_path)

    img = Image.open(img_path)
    cropped_img = crop_resize(img, get_latlong(bbox_info))
    predicted_text = detector.predict(cropped_img)
    # cropped_img.show()

    imgTk = ImageTk.PhotoImage(cropped_img)
    label_img.config(image=imgTk)
    label_img.image = imgTk
    label_img_name.config(text=f'Img: {img_name}')
    entry_noidung.delete(0, tk.END)
    entry_noidung.insert(0, predicted_text)
    label_bbox.config(text=display_bbox_info(bbox_info))

    current_index = int(bbox_info[0])
    with open(resume_index_filepath, 'w') as f:
        f.write(str(current_index))


nextbbox_frame = tk.Frame(window, width=600, height=200)
nextbbox_frame.pack()

button_next_item = tk.Button(nextbbox_frame, text='Next', command=next_bbox)
button_next_item.pack(side='bottom', pady=(50, 0))

window.mainloop()