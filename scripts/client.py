import sys
import requests
import json


def main():
    if len(sys.argv) == 3:
        resp = requests.post("http://localhost:5000/check",
                             data={
                                 "image_path": sys.argv[1],
                                 "output_path": sys.argv[2]
                             }
                             )
        print(json.dumps(resp.json()))
    else:
        print('Invalid argv')


if __name__ == '__main__':
    main()
