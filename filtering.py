n = int(input("Enter the size of the input image: "))
k = int(input("Enter the size of the kernel image: "))
img=[]
for i in range(0,n):
    img1=[]
    for j in range(0,n):
        ele = int(input("Enter the value: "))
        img1.append(ele)
    img.append((img1))
print(img[0][0])
kernel =[]
for i in range(0,k):
    kernel1=[]
    for j in range(0,k):
        ele = int(input("Enter the value: "))
        kernel1.append(ele)
    kernel.append((kernel1))
output=[]
for k in range(0,n):
    output1=[]
    sum=0
    for i in range(0,n):
        for j in range(0,k):
            sum+=kernel[i][j]*img[i][j]
        output1.append(sum)
    output.append((output1))
print(output)