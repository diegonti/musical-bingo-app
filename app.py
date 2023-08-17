from flask import Flask, render_template, request, send_file
import os
import shutil
import random
import musical_bingo
# import your_image_generation_script

def create_upload_dirs(folder,seed):
    
    try: os.mkdir("uploads")
    except FileExistsError: pass

    try: os.mkdir(f"uploads/{folder}/")
    except FileExistsError: pass

    try: os.mkdir(f"uploads/{folder}/{seed}")
    except FileExistsError: pass


def store_upload_images(uploaded_images,out_folder):
    # Store uploaded images temporarily
    image_paths = []
    for image in uploaded_images:
        image_path = out_folder + image.filename
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths

def store_upload_songs(uploaded_songs,out_folder):
    # Store uploaded songs temporarily
    uploaded_songs = uploaded_songs[0]
    uploaded_songs_path = out_folder + "songs.txt"
    uploaded_songs.save(uploaded_songs_path)
    return uploaded_songs_path

def parse_parameters(parameters:dict):

    for key,value in list(parameters.items()):
        if value=="": del parameters[key]

    return parameters

    

app = Flask(__name__,static_folder="./static/")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_images', methods=['POST'])
def generate_images():

    user_seed = random.randint(1,10000)

    # Get uploaded images and songs
    uploaded_images = request.files.getlist('images')
    uploaded_songs = request.files.getlist('songs')

    parameters = request.form.copy()
    parameters = parse_parameters(parameters)

    IMG_FOLDER = f"uploads/img/{user_seed}/"
    SONGS_FOLDER = f"uploads/songs/{user_seed}/"

    parameters["IMG_FOLDER"] = IMG_FOLDER
    parameters["SONGS_FILE"] = SONGS_FOLDER + "songs.txt"

    # Store images
    create_upload_dirs("img",user_seed)
    if not uploaded_images[0].filename == "": 
        store_upload_images(uploaded_images, IMG_FOLDER)
    else: parameters["N_IMAGES_CARD"] = 0

    # Store songs
    create_upload_dirs("songs",user_seed)
    if not uploaded_songs[0].filename == "":
        store_upload_songs(uploaded_songs, SONGS_FOLDER)
    else: parameters["SONGS_FILE"] = "./static/numbers.txt"

    # Call your image generation script with image_paths
    musical_bingo.main(parameters)
    shutil.make_archive("bills","zip","bills/")

    shutil.rmtree(IMG_FOLDER)
    shutil.rmtree(SONGS_FOLDER)


    return send_file("bills.zip", as_attachment=True, download_name="bills.zip")
    # return render_template('result.html', images=generated_images_paths)
    # return render_template('results.html')


@app.route('/download_images')
def download_images():
    # Code to zip and save generated images
    # Return the zip file as a downloadable attachment
    return send_file("bills.zip", as_attachment=True, download_name="bills.zip")

if __name__ == '__main__':
    app.run()
