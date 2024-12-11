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


def get_all_data(id):
    ids = read_id_file()
    [email] = [k for k, v in ids.items() if v == id]
    user_dir = get_user_dir(email)
    data_dir = path.join(user_dir, 'assessment_data')
    data = os.listdir(data_dir)
    if not len(data):
        return False
    else:
        return True

