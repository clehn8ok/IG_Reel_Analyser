i've written a script, where you can load a csv file from someones IG Reel export  ( done with the chrome extension Sort Feed | App for Instagram™) (your IG must be set to english to work)



This returns a csv of the most viral reels.



Then i have a python script, which generates for each video a pdf with screenshots, for each jumpcut, with timestamps and duration.



it also generates a new statistics CSV with

Avg Clip duration 

median clip duration

likes/view

comments/view





those python packages are required:

pip install yt-dlp

pip install opencv-python

pip install reportlab

pip install numpy



so you need:

the chrome extension

a python ide

the req. python packages

a csv export from the chrome extension



then you run the python script and it generates for each reel over 100k views a pdf with screenshots and durations for each clip out of that reel. + some addional statstic



