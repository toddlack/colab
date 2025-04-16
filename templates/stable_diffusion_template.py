# Code for initializing download of Stable Diffusion models and web UI in colab
from calendar import c
import json
import os, sys, subprocess, time, glob
from sympy import root
import torch
from google.colab import drive, userdata, files

from IPython.display import clear_output

#dictionary of model paths
model_dict = {
    'sd1.5': 'https://huggingface.co/stabilityai/stable-diffusion-1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
    'sd2.1': 'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/sd-v1-4-full-ema.ckpt',
    'flux1_dev' : 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors',
    'dreamshaper': 'https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors',
    'f222': 'https://huggingface.co/acheong08/f222/resolve/main/f222.ckpt',
    'realistic_vision': 'https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors',
}

def getColabSecretFromUserData(user_Data=None):
  if user_Data:
    secret = userdata.get(user_Data)
  else:
    secret = None
  return secret


def clear():
    return clear_output()

def aria2c(url=None, filename=None, destinationDir = None):
  """
  Downloads a file from a given URL using aria2c.
  
  Parameters
  ----------
  url : str
    URL of the file to download.
  filename : str, optional
    Name of the output file. If not provided, the filename will be the last part of the URL.
  destinationDir : str, optional
    Directory to save the downloaded file. If not provided, the file will be saved in the current directory.
  
  Returns
  -------
  subprocess.CompletedProcess
    The result of the aria2c command.
  """
  
  ariaParams = ['aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M', url, '-o', filename]
  if destinationDir: 
    ariaParams.append('--dir')
    ariaParams.append(destinationDir)

  return subprocess.run(ariaParams)

def downLoadModel(url, outfile = None, destinationDir = None):
  """
  Downloads a model from Hugging Face or Civitai using aria2c.

  Parameters
  ----------
  url : str
    The URL of the model to download.
  outfile : str, optional
    The name of the downloaded file. Defaults to the filename from the URL.
  destinationDir : str, optional
    The directory where the model will be downloaded. Defaults to the current working directory.

  Notes
  -----
  For Hugging Face models, the `outfile` parameter is optional and will default to the filename from the URL.
  For Civitai models, the `outfile` parameter is required and the `destinationDir` parameter is ignored.
  """
 
  if destinationDir:
    destination = f' --dir {destinationDir}'
  if 'huggingface.co' in url:
    if outfile:
      filename = outfile
    else:
      filename = url.split('/')[-1]
      filename = filename.removesuffix('?download=true')
    #run comman aria2c
    aria2c(url, filename, destinationDir)
    
  else:
    # civitai - adjust url accordingly with civitai_api_key
    if not Civitai_API_Key:
      Civitai_API_Key = getColabSecretFromUserData('CIVITAI_API_KEY')
    qp_separator = '?' if '?' in url else '&'
    civitai_url = f"{url}{qp_separator}token={Civitai_API_Key}"
    if outfile:
      aria2c(civitai_url, outfile, destinationDir)
    else:
      aria2c(civitai_url, url.split('/')[-1], destinationDir)

def downloadAllModels():
  os.chdir(f"{root}/stable-diffusion-webui-forge/models/Stable-diffusion")
  print('⏳ Downloading models ...')
  # Flux
  if Flux1_dev:
    downLoadModel(model_dict['flux1_dev'], 'flux1-dev.safetensors')
  # Dreamshaper
  if Dreamshaper:
    downLoadModel(model_dict['dreamshaper'], 'DreamShaper_8_pruned.safetensors')
  # f222
  if f222:
    downLoadModel(model_dict['f222'], 'f222.ckpt')
  # Realistic Vision
  if Realistic_Vision:
    downLoadModel(model_dict['realistic_vision'], 'Realistic_Vision_V5.1_fp16-no-ema.safetensors')

def link_files(source, dest):
    '''
    Create symlinks for all files in the source folder to the dest folder

    Parameters
    ----------
    source : str
        Absolute path of the source folder
    dest : str
        Absolute path of the destination folder

    Notes
    -----
    This function creates symlinks for all files in the source folder to the dest folder.
    If the source folder does not exist, it will be created.
    If the dest folder does not exist, it will be created.
    '''
    if not os.path.exists(source):
        subprocess.run(['mkdir', '-p', source])
    if not os.path.exists(dest):
        subprocess.run(['mkdir', '-p', dest])
    model_files = glob.glob(source + '/*')
    os.chdir(dest)
    for f in model_files:
        print(f'Linking model {f} in {dest}')
        os.symlink(f, os.path.basename(f))

def initConfigFile(symlink: bool = True, makedir: bool = False, sourceDir: str = '', destDir: str = '', file: str = '') -> None:
    """
    Initialize the config files for the stable diffusion webui.

    Parameters
    ----------
    symlink : bool, optional
        Whether to create symlinks for the config files. The default is True.
    makedir : bool, optional
        Whether to create a directory if the file does not exist. The default is False.
    sourceDir : str, optional
        The source directory where the config files are located. The default is ''.
        This is typically your Google Drive path. Permanent files go here, and the destination directory is the webui folder.
    destDir : str, optional
        The destination directory where the config files will be linked to. The default is ''.
    file : str, optional
        The name of the config file to link. The default is ''.

    Returns
    -------
    None

    Notes
    -----
    This function will create symlinks for the config files in the source directory to the destination directory.
    If the source directory does not exist, it will be created.
    If the dest directory does not exist, it will be created.
    """
    # Check if the file exists in the source directory
    if not os.path.exists(f'{sourceDir}/{file}'):
        print(f'{file} not found in {sourceDir}. Creating empty file.')
        # Create directory or file based on the makedir flag
        if makedir:
            os.makedirs(f'{sourceDir}/{file}')
        else:
            with open(f'{sourceDir}/{file}', 'w') as fp:
                pass

    # Create symlink if the symlink flag is set
    if symlink:
        link_files(f'{sourceDir}/{file}', f'{destDir}/{file}')


def gitClone(url=None, destination=None):
    subprocess.run(['git', 'clone', url, destination])

### main section - options first, then execution

#@markdown ## Setup options
#@markdown ### output and webui options
output_path = 'AI_PICS' #@param {type:"string"}
version='' #@param {type:"string"}
username = 'a' #@param {type:"string"}
password = 'a' #@param {type:"string"}
Civitai_API_Key = 'cd1370c28e6697428dc380d19dc5b678' #@param {type: "string"}
#@markdown Create API key [here](https://civitai.com/user/account). You can also use secrets.

#@markdown ### Only check the models you are going to use:
Flux1_dev = True #@param {type:"boolean"}   
Dreamshaper = True #@param {type:"boolean"}
f222 = False #@param {type:"boolean"}
Realistic_Vision = False #@param {type:"boolean"}

#@markdown ### ControlNet models:
SD_1_5_ControlNet_models = True #@param{type: "boolean"}
SDXL_ControlNet_models = False #@param{type: "boolean"}
IP_Adapter_models = True #@param{type: "boolean"}

#@markdown ### Install extensions from URL (separate them with comma).
Extensions_from_URL = 'https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper,https://github.com/Gourieff/sd-webui-reactor,https://github.com/civitai/sd_civitai_extension.git' #@param {type: "string"}
#@markdown ### Extra Web-UI arguments
Extra_arguments = ' --theme dark' #@param {type: "string"}
sd_next_git_url ='https://github.com/vladmandic/sdnext.git'


#Connect to google drive
root = '/content'
webui_root=f'{root}/sd_next'
drive.mount(f'{root}/drive')
user_output_path = f"{root}/drive/MyDrive/{output_path}"
#Clone Webui - could be forge, automatic1111 or sd.next
print('⏳ Cloning Stable Diffusion WebUI Forge ...')
gitClone(sd_next_git_url, f"{root}/sd_next")
#apt-get -y install -qq aria2
subprocess.run(['apt-get', 'install', '-y', 'aria2'], check=True)

os.chdir(webui_root)
python_args = ['python', f'{root}/sd_next/launch.py']
python_args.extend(Extra_arguments.split(' '))

result = subprocess.run(python_args, capture_output=True, text=True)
print(result.stdout)

