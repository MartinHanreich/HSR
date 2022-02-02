# HSR
Human Speaker Replacement in videos

# Overview
*Human Speaker Replacement* is about creating **Face Swapping** in videos without the use of deep learning or any other similar approach. Instead it uses simple calculations and freely available tools to create a similar effect while requiring less resources and data.

# How to use it
Clone the repo and open the HumanSpeakerReplacement.py and the HSRProcessing.py scripts with an environement of your choice and install the necessary requirement. In the *HSRProcessing.py* script specifiy the name of the video file to use for the replacement. There are already videos provided in the **videos** folder. If **only images** should be used copy them into the **imgs** folder delete other images in there which should not be used and set the *FROM_VIDEO* variable to *False* before executing. In the **HumanSpeakerReplacement.py** script specify the name of the video where the face should be replaced. Afterwards execute it. A video file is then created named *human_speaker_replacement.mp4*. 
