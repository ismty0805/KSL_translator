import cv2, os

def flip_images():
	gest_folder = "gestures"
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):
		if(g_id == "7" or g_id == "15_real"):
			for i in range(1200):
				path = gest_folder+"/"+g_id+"/"+str(i+1)+".jpg"
				new_path = gest_folder+"/"+g_id+"/"+str(i+1+1200)+".jpg"
				print(path)
				img = cv2.imread(path, 0)
				img = cv2.flip(img, 1)
				cv2.imwrite(new_path, img)

flip_images()
