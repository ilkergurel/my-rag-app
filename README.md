git init

#After that create Gitlab project in web and push the files to repo

#Install requirements
pip install -r requirements.txt

#If GPU utilization
conda install pytorch torchvision torchaudio cudatoolkit=12.6 cudnn -c pytorch -c nvidia
