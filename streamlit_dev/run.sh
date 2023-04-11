docker run -it\
 -v "$(pwd):/home/app"\
 -e PORT=80\
 -p 4001:80\
 streamlit