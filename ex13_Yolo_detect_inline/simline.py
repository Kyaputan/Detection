import matplotlib.pyplot as plt

# กำหนดจุดเริ่มต้นและจุดสิ้นสุดของเส้น
x_coords = [480, 480]
y_coords = [0, 540]
region_points = [(480, 480), (0, 540)]
# วาดเส้น
plt.plot(x_coords, y_coords, color='blue')

# ตั้งค่าขอบเขตของกราฟ
plt.xlim(0, 960)
plt.ylim(0, 540)

# แสดงกราฟ
plt.show()
