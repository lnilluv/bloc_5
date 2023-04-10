docker build -t getaround-streamlit .

docker run -it\
  -p 4001:4001\
  -v "$(pwd):/app"\
  -e PORT=4001\
  getaround-streamlit