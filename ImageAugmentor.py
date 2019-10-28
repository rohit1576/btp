import Augmentor
p = Augmentor.Pipeline("/Users/kanishkverma/Desktop/btp/dataset/volume_up")
p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
p.sample(3000)
p.process()
