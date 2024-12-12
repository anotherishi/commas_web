# Setup instructions

(assuming linux)

1. Create a virtual environment\
`python3 -m venv venv`

2. Activate the environment\
`source ./venv/bin/activate`

3. Install requirements\
`pip install -r requirements.txt`

4. Start the server (production server using waitress)\
`waitress-serve main:server`

5. Open the address in your browser
