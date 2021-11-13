def kNN(x, y, k):
	yield x + k, y + k
	yield x + k, y - k
	yield x - k, y + k
	yield x - k, y - k
	yield x + k, y
	yield x - k, y
	yield x, y + k
	yield x, y - k


class Board:
	OUT = -1
	BLANK = 0
	P1 = 1
	P2 = 2

	def __init__(self, width, height, k=2, board=None):  # k: kNN
		self.p1 = dict()
		self.p2 = dict()
		self.width = width
		self.height = height
		self.k = k

		# Make players
		if board:
			for x in range(self.width):
				for y in range(self.height):
					if board[x][y] == 1:
						self.p1[(x, y)] = 1  # Assume every P1 chess has at least 1 point
						for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
							i = 1
							while i:
								col, val = self[(x+i*dx, y+i*dy)]
								if col == self.P1:
									i += 1
									self.p1[(x+i*dx, y+i*dy)] += 1
									self.p1[(x, y)] += 1
								else:
									i = 0
							i = -1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P1:
									i -= 1
									self.p1[(x+i*dx, y+i*dy)] += 1
									self.p1[(x, y)] += 1
								else:
									i = 0
							i = 1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P2:
									i += 1
									self.p2[(x + i * dx, y + i * dy)] = min(-1, self.p2[(x + i * dx, y + i * dy)]+1)
								else:
									i = 0
							i = -1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P2:
									i -= 1
									self.p2[(x + i * dx, y + i * dy)] = min(-1, self.p2[(x + i * dx, y + i * dy)]+1)
								else:
									i = 0

					elif board[x][y] == 2:
						self.p2[(x, y)] = -1  # Assume every P2 chess has at most -1 point
						for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
							i = 1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P2:
									i += 1
									self.p2[(x + i * dx, y + i * dy)] -= 1
									self.p2[(x, y)] -= 1
								else:
									i = 0
							i = -1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P2:
									i -= 1
									self.p2[(x + i * dx, y + i * dy)] -= 1
									self.p2[(x, y)] -= 1
								else:
									i = 0
							i = 1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P1:
									i += 1
									self.p1[(x + i * dx, y + i * dy)] = max(1, self.p1[(x + i * dx, y + i * dy)] - 1)
								else:
									i = 0
							i = -1
							while i:
								col, val = self[(x + i * dx, y + i * dy)]
								if col == self.P1:
									i -= 1
									self.p1[(x + i * dx, y + i * dy)] = max(1, self.p1[(x + i * dx, y + i * dy)] - 1)
								else:
									i = 0

		# Make frontier
		self.frontier = set()
		for x, y in self.p1:
			for i in range(1, k + 1):
				for neighbour in kNN(x, y, i):
					if self[neighbour][0] == 0:
						self.frontier.add(neighbour)
		for x, y in self.p2:
			for i in range(1, k + 1):
				for neighbour in kNN(x, y, i):
					if self[neighbour][0] == 0:
						self.frontier.add(neighbour)

	def __str__(self):
		return f"Player 1: {self.p1}\n" \
				f"Player 2: {self.p2}\n" \
				f"Frontier: {self.frontier}"

	def __repr__(self):
		return f"Player 1: {self.p1}\n" \
				f"Player 2: {self.p2}\n" \
				f"Frontier: {self.frontier}"

	def __getitem__(self, index):
		"""
		ALERT: Index by board[(x, y)] now !!!
		0: isFree
		1: isP1
		2: isP2
		-1: NOT VALID
		"""
		if self.isP1(index):
			return 1, self.p1[index]
		elif self.isP2(index):
			return 2, self.p2[index]
		elif self.isValid(index):
			return 0, None
		return -1, None

	def isP1(self, index):
		return index in self.p1

	def isP2(self, index):
		return index in self.p2

	def isValid(self, index):
		x, y = index
		return 0 <= x < self.width and 0 <= y < self.height

	def addP1(self, index):
		"""
		If the addition is temporary, we have to copy the previous frontier.
		If the addition is permanent, then do it without a copy.
		"""
		assert self[index][0] == 0, f"Error occurs trying to play P1 at {index}"
		x, y = index

		self.p1[(x, y)] = 1  # Assume every P1 chess has at least 1 point
		for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
			i = 1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P1:
					self.p1[(x + i * dx, y + i * dy)] += 1
					i += 1
				else:
					i = 0
			i = -1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P1:
					self.p1[(x + i * dx, y + i * dy)] += 1
					i -= 1
				else:
					i = 0
			i = 1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P2:
					self.p2[(x + i * dx, y + i * dy)] = min(-1, self.p2[(x + i * dx, y + i * dy)] + 1)
					i += 1
				else:
					i = 0
			i = -1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P2:
					self.p2[(x + i * dx, y + i * dy)] = min(-1, self.p2[(x + i * dx, y + i * dy)] + 1)
					i -= 1
				else:
					i = 0

		if (x, y) in self.frontier:
			self.frontier.remove((x, y))
		for i in range(1, self.k + 1):
			for neighbour in kNN(x, y, i):
				if self[neighbour][0] == 0:
					self.frontier.add(neighbour)

	def addP2(self, index):
		"""
		If the addition is temporary, we have to copy the previous frontier.
		If the addition is permanent, then do it without a copy.
		"""
		assert self[index][0] == 0, f"Error occurs trying to play P2 at {index}"
		x, y = index

		self.p2[(x, y)] = -1  # Assume every P2 chess has at most -1 point
		for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
			i = 1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P2:
					self.p2[(x + i * dx, y + i * dy)] -= 1
					i += 1
				else:
					i = 0
			i = -1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P2:
					self.p2[(x + i * dx, y + i * dy)] -= 1
					i -= 1
				else:
					i = 0
			i = 1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P1:
					self.p1[(x + i * dx, y + i * dy)] = max(1, self.p1[(x + i * dx, y + i * dy)] - 1)
					i += 1
				else:
					i = 0
			i = -1
			while i:
				col, val = self[(x + i * dx, y + i * dy)]
				if col == self.P1:
					self.p1[(x + i * dx, y + i * dy)] = max(1, self.p1[(x + i * dx, y + i * dy)] - 1)
					i -= 1
				else:
					i = 0

		if (x, y) in self.frontier:
			self.frontier.remove((x, y))
		for i in range(1, self.k + 1):
			for neighbour in kNN(x, y, i):
				if self[neighbour][0] == 0:
					self.frontier.add(neighbour)

# 	FIXME: Do we still need remove? Since we can do deepcopy here.
	# def rmP1(self, index, frontier):
	# 	assert self[index] == 1, f"Error occurs trying to remove P1 at {index}"
	# 	self.frontier = frontier
	# 	self.p1.remove(index)
	#
	# def rmP2(self, index, frontier):
	# 	assert self[index] == 2, f"Error occurs trying to remove P2 at {index}"
	# 	self.frontier = frontier
	# 	self.p2.remove(index)

	def display(self):
		# TODO: Visualize it
		print(f'  {" ".join((str(i) for i in range(self.width)))}')
		display = [['-' for i in range(self.width)] for j in range(self.height)]
		for x, y in self.p1.keys():
			display[x][y] = '1'
		for x, y in self.p2.keys():
			display[x][y] = '2'
		for x, y in self.frontier:
			display[x][y] = 'f'
		for i, row in enumerate(display):
			print(f'{i} {" ".join(row)}')

	def displayEval(self):
		# TODO: Visualize it
		print(f'  {" ".join((str(i) for i in range(self.width)))}')
		display = [['-' for i in range(self.width)] for j in range(self.height)]
		for x, y in self.p1.keys():
			display[x][y] = str(self.p1[(x, y)])
		for x, y in self.p2.keys():
			display[x][y] = str(self.p2[(x, y)])
		for i, row in enumerate(display):
			print(f'{i} {" ".join(row)}')

	def utility(self, player):
		# evaluate the self.threat
		if player == 1:
			pass
		if player == 2:
			pass

	def clear(self):
		self.p1 = dict()
		self.p2 = dict()
		self.frontier = set()


if __name__ == '__main__':
	pass