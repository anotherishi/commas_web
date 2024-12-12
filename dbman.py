import json
import os, os.path as path
from uuid import uuid4


def uuid():
    return f"{uuid4()}{uuid4()}".replace("-", "")


db_path = path.join(os.curdir, "db")


def user_exists(email):
    user_dir = path.join(db_path, email)
    return path.isdir(user_dir)


def get_user_dir(email):
    return path.join(db_path, email)


def create_user(data):
    details = dict(data)
    uid = str(uuid())

    ids = read_id_file()
    ids[details["email"]] = uid
    update_id_file(ids)
    user_dir = get_user_dir(details["email"])
    os.mkdir(user_dir)
    os.mkdir(path.join(user_dir, "assessment_data"))
    with open(path.join(user_dir, "details"), "w") as detials_file:
        json.dump(details, detials_file)
    return uid


def read_id_file():
    with open(path.join(db_path, "ids")) as id_file:
        return json.load(id_file)


def update_id_file(ids):
    with open(path.join(db_path, "ids"), "w") as id_file:
        json.dump(ids, id_file)


def session_id_exists(id):
    print(id, read_id_file().values())
    return id in read_id_file().values()


def check_login_request(data):
    email = data["email"]
    password = data["password"]
    user_dir = get_user_dir(email)
    with open(path.join(user_dir, "details")) as details_file:
        details = json.load(details_file)
        return email == details["email"] and password == details["password"]


def init_login(data):
    email = data["email"]
    ids = read_id_file()
    uid_new = uuid()
    ids.update({email: uid_new})
    update_id_file(ids)
    return uid_new

def get_email_from_id(id):
    ids = read_id_file()
    return [k for k, v in ids.items() if v == id][0]

def get_data_dir_from_id(id):
    ids = read_id_file()
    [email] = [k for k, v in ids.items() if v == id]
    user_dir = get_user_dir(email)
    return path.join(user_dir, 'assessment_data')


def get_all_data(id):
    data_dir = get_data_dir_from_id(id)
    length = len(os.listdir(data_dir))
    if not length:
        return False
    else:
        data = {}
        for i in range(length):
            i = str(i)
            data[i] = {}
            with open(path.join(data_dir, i, "details")) as details_file:
                data[i]["details"] = json.loads(details_file.read())
            with open(path.join(data_dir, i, "results")) as results_file:
                data[i]["results"] = json.loads(results_file.read())
        return data


def handle_upload(cookie, video_file, audio_file, metadata):
    data_dir = get_data_dir_from_id(cookie)
    n = str(len(os.listdir(data_dir)))
    os.mkdir(path.join(data_dir, n))
    video_filename = path.join(data_dir,n, video_file.filename)
    audio_filename = path.join(data_dir,n, audio_file.filename)
    video_file.save(video_filename)
    audio_file.save(audio_filename)
    metadata = json.loads(metadata)
    metadata["n"] = n
    browser_transcript = metadata.get('transcript')
    with open(path.join(data_dir,n, "details"), 'w') as details_file:
        json.dump(metadata, details_file)
    with open(path.join(data_dir,n, "transcript"), 'w') as transcript_file:
        transcript_file.write(browser_transcript)
    
    # now audio, video, transcript files are saved
    # pass to model for processing
    # save results

    results = {}

    with open(path.join(data_dir,n, "results"), 'w') as result_file:
        json.dump(results, result_file)
    return n

def get_details(id, n):
    data_dir = get_data_dir_from_id(id)
    with open(path.join(data_dir, n, 'details')) as details_file:
        return json.load(details_file)