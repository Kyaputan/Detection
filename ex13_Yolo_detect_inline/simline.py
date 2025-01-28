import matplotlib.pyplot as plt

# กำหนดจุดเริ่มต้นและจุดสิ้นสุดของเส้น
x_coords = [200, 900]
y_coords = [200, 200]

# วาดเส้น
plt.plot(x_coords, y_coords, color='blue')

# ตั้งค่าขอบเขตของกราฟ
plt.xlim(0, 1100)
plt.ylim(0, 500)

# แสดงกราฟ
plt.show()
