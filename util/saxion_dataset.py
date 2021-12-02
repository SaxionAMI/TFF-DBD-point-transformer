import numpy as np
import h5py
import scipy.stats as stats
from scipy.spatial.transform import Rotation as R
import pylas
import torch

class SaxionDataset(torch.utils.data.IterableDataset):
		# Normalise to unit circle
		def pc_normalise(self, pc):
				#centroid = np.mean(pc, axis=0)
				vmin = np.min(pc, axis=0)
				vmax = np.max(pc, axis=0)
				span = vmax-vmin
				centroid = vmin + span/2
				#centroid[2] = 0 # Leave z-axis alone
				#print(centroid)

				pc = pc - centroid
				#m = np.max(np.linalg.norm(pc, axis=1))
				m = 13.5 # 27m diagonal range will be from -1 to 1
				pc = pc / m
				return pc

		def jitter(self, pc):
				pc = pc + self.jitter_src.rvs(size=pc.shape)
				return pc

		def shuffle(self, points, labels):
				perms = np.random.permutation(points.shape[0])
				return points[perms,:], labels[perms]

		def translate(self, pc):
				#mint = 10/2700
				#maxt = 150/2700
				zrange = 100/2700

				#minmax = np.zeros((2,2))
				#minmax[:,0] = -1 - np.min(pc, axis=0)[0:2]
				#minmax[:,1] =  1 - np.max(pc, axis=0)[0:2]
				#minmax = np.abs(minmax)
				#delta = minmax
				#delta[delta<mint] = 0
				#delta[delta>maxt] = maxt
				#dx = (delta[1,0]+delta[0,0])*self.rng.random() - delta[0,0]
				#dy = (delta[1,1]+delta[0,1])*self.rng.random() - delta[0,1]
				#dz = np.random.uniform(-zrange,zrange)
				##dz = 0
				#pc = pc+[dx,dy,dz]
				##print("dx: {:.2f} dy: {:.2f}".format(dx,dy))
				dz = np.random.uniform(-zrange,zrange,size=(1,3))
				pc = pc + dz
				return pc

		# Rotates objects randomly on the x,y-plane
		def rotate(self, pc):
				#rotmat = R.random().as_matrix()
				#pc = np.matmul(pc,rotmat)
				#a = np.deg2rad(-89)
				#b = -a
				#delta_theta = (b-a)*self.rng.random() + a
				delta_theta = np.random.uniform(-180,180)
				delta_theta = np.deg2rad(delta_theta)

				# Create rotation matrix
				rotmat = np.zeros((2,2))
				rotmat[0,0] = np.cos(delta_theta)
				rotmat[0,1] = -np.sin(delta_theta)
				rotmat[1,0] = np.sin(delta_theta)
				rotmat[1,1] = np.cos(delta_theta)

				# Rotate x,y points
				pc[:,0:2] = np.matmul(pc[:,0:2],rotmat)
				# Rotate x,y normals
				pc[:,3:5] = np.matmul(pc[:,3:5],rotmat)
				return pc;


		def __init__(self, path, permute=False, lo=None, mode=None):
				self.path = path
				self.permute = permute
				self.cnt = 0
				self.val_cnt = 0
				self.npoints = 0
				self.lo = lo
				self.mode = mode

				# Jitter src
				lo,hi = -5,5
				mu, sigma = 0,2
				lo = lo/2700
				hi = hi/2700
				mu = mu/2700
				sigma = sigma/2700
				self.jitter_src = stats.truncnorm((lo-mu)/sigma, (hi-mu)/sigma, loc=mu, scale=sigma)

				self.rng = np.random.default_rng(42)
				
				if path is None:
					return

				#load data
				print(path)
				with h5py.File(path, 'r')as hf:
						print(hf.keys())
						self.data = hf['data'][()]
						self.labels = hf['labels'][()]

				# Normalise
				for i in np.arange(self.data.shape[0]):
						self.data[i,:,0:3] = self.pc_normalise(self.data[i,:,0:3])

				data_labels = np.array([0,1,2,3,4,5,6,7,8,9,10,14,15,16])
				for idx,v in enumerate(data_labels):
					self.labels[self.labels==v] = idx

				print("Label max: {:d}, label min: {:d}".format(np.max(self.labels), np.min(self.labels)))

				self.npoints = self.data.shape[1]
				#self.weights = np.ones((14,1))
				#self.weights[0,0] = 0
				self.weights = self.get_class_weights()

		def __len__(self):
			return self.data.shape[0]

		# Watchout: hard coded number of classes!!! TODO
		# Class weights based on entire dataset, exluding the left out samples
		def get_class_weights(self):
			edges = np.arange(15)
			mask = np.ones(self.__len__(), np.bool)
			if self.lo is not None:
				mask[self.lo] = False
			hist,_ = np.histogram(self.labels[mask,:], edges)
			hist[0] = 0 # Give unlabbeld class weight zero
			hist = hist/np.sum(hist)
			hist = (1 - hist)/12
			hist[0] = 0 # Give unlabbeld class weight zero
			return hist

		def __iter__(self):
				return self

		def __call__(self):
				return self

		def __getitem__(self, key):
			labels = self.labels[key,:]
			labels = np.expand_dims(labels, axis=1)
			points = self.data[key,:,:]
			#return (self.data[key,:,:], labels, self.weights)
			#return (points, labels)
			feats = np.hstack((np.ones((points.shape[0],1)),points[:,0:3]))
			return (points[:,0:3], feats ,labels)
		
		# Returns a sample
		def __next__(self):
			if self.cnt >= self.__len__():
				self.cnt = 0
				#if self.mode != "val":
				#	raise StopIteration

			# Check if next one is a leave-out
			if self.lo is not None:
				while self.cnt in self.lo:
					self.cnt = self.cnt+1
					if self.cnt >= self.__len__():
						self.cnt = 0
			points = self.data[self.cnt,...]
			#feats = self.data[self.cnt,:,3:]

			labels = self.labels[self.cnt,:]

			# Apply permutations
			if self.permute:
				#points, labels = self.shuffle(points, labels)
				points[:,0:3] = self.jitter(points[:,0:3])
				points = self.rotate(points)
				points[:,0:3] = self.translate(points[:,0:3])
			self.cnt = self.cnt+1

            #https://github.com/ultralytics/yolov3/issues/249
			weights = np.bincount(labels, minlength=14)
			weights[0] = 0
			weights = np.divide(np.sum(weights),weights, out=np.zeros_like(weights, dtype=float), where=(weights!=0))
			weights = np.take(weights, labels)

			labels = np.expand_dims(labels, axis=1)
			feats = np.hstack((np.ones((points.shape[0],1)),points[:,0:3]))
			coord = torch.FloatTensor(points[:,0:3])
			feat = torch.FloatTensor(feats)
			label = torch.LongTensor(labels)
			offset = torch.IntTensor([coord.shape[0]])
			weights = torch.FloatTensor(weights)
			return coord, feat, label, offset, weights

if __name__ == '__main__':
		d = SaxionDataset('train_131072.hdf5', permute=True, lo=[7])
		print(d.weights.shape)
		print(d.get_class_weights())

		#dg = Dataset.from_generator(d, (tf.float32, tf.uint8, tf.float32), ((d.npoints,3),(d.npoints,1),(d.npoints,1)))
		#dg = dg.batch(14, drop_remainder=True)
		#print(dg)
		#a = next(iter(dg))
		#print(a[0][0][0:5,:])
		#la = a[0][0].numpy()
		#print(type(la))

		#x,y = d[7]
		#x = np.expand_dims(x, axis=0)
		#y = np.expand_dims(y, axis=0)
		#print(x.shape)
		#print(y.shape)
		#pc = x[0,...]
		#l = y 

		##print(a)
		#for i in np.arange(len(d)):
		#	pc = a[0][i][...].numpy()
		#	print(pc.shape)
		#	l = a[1][i].numpy()
		#	
		#	#pc,l = d[i]
		#	las = pylas.create(point_format_id=3)
		#	las.add_extra_dim(name="class", type="uint8")
		#	las.x = pc[:,0]
		#	las.y = pc[:,1]
		#	las.z = pc[:,2]
		#	las['class'] = np.squeeze(l)
		#	las.write("batch_{:d}.laz".format(i))
		#	print("*******************")

		#	las = pylas.create(point_format_id=3)
		#	las.add_extra_dim(name="class", type="uint8")
		#	las.x = pc[:,0]
		#	las.y = pc[:,1]
		#	las.z = pc[:,2]
		#	las['class'] = np.squeeze(l)
		#	las.write("{:d}_permuts.laz".format(i))
		#print(l[0:10])
