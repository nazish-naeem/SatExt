import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
fig, ax = plt.subplots()
ax.axis('off')
ims = []
path = '/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement/inference_tests/infer_20240731-030732/240000'
hr_img = cv2.imread(path+'/2_1_hr.png')
hr_img = hr_img[...,::-1]
for i in range(1,11):
    sr_img = cv2.imread(path+'/2_'+str(i)+'_sr.png')
    sr_img = sr_img[...,::-1]
    

    final_img = np.zeros_like(sr_img).astype(float)
    final_img[:,:,0] = sr_img[:,:,0].astype(float)-(np.mean(sr_img[:,:,0])-np.mean(hr_img[:,:,0]))
    final_img[:,:,1] = sr_img[:,:,1].astype(float)-(np.mean(sr_img[:,:,1])-np.mean(hr_img[:,:,1]))
    final_img[:,:,2] = sr_img[:,:,2].astype(float)-(np.mean(sr_img[:,:,2])-np.mean(hr_img[:,:,2]))

    cv2.imwrite(path+'/2_'+str(i)+'_color_correction.png',(np.clip(final_img[...,::-1],0,255)).astype(int))
    xx=(np.clip(final_img,0,255)).astype(int)
    print(xx.shape)
    im = ax.imshow(xx)
    # im = cv2.imshow('image',xx)
    if i == 1:
        # im = cv2.imshow('image',xx)
        im = ax.imshow((np.clip(final_img,0,255)).astype(int))  # show an initial one first
    ims.append([im])

path = '/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement/final_img.png'
sr_img = cv2.imread(path)

# cv2.imwrite(path+'/2_'+str(i)+'_color_correction.png',(np.clip(final_img[...,::-1],0,255)).astype(int))
# xx=(np.clip(final_img,0,255)).astype(int)
im = ax.imshow(sr_img[...,::-1])
# im = cv2.imshow('image',xx)
ims.append([im])







ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=10000)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
writer = animation.FFMpegWriter(
    fps=5, metadata=dict(artist='Me'), bitrate=1800)
ani.save("movie.mp4", writer=writer)

plt.show()