text_tsir = '''
Armadillo RLIT[1] @ PSNR: 25.907
Armadillo RLIT[1] @ SSIM: 0.9359
Armadillo RLIT[1] @ LPIPS: 0.0665
Armadillo RLIT[2] @ PSNR: 22.628
Armadillo RLIT[2] @ SSIM: 0.9179
Armadillo RLIT[2] @ LPIPS: 0.0791
Armadillo RLIT[3] @ PSNR: 27.014
Armadillo RLIT[3] @ SSIM: 0.9326
Armadillo RLIT[3] @ LPIPS: 0.0571
Armadillo RLIT[4] @ PSNR: 26.171
Armadillo RLIT[4] @ SSIM: 0.9299
Armadillo RLIT[4] @ LPIPS: 0.0690
Armadillo RLIT[5] @ PSNR: 29.816
Armadillo RLIT[5] @ SSIM: 0.9512
Armadillo RLIT[5] @ LPIPS: 0.0512
Armadillo Albedo @ PSNR: 30.398
Armadillo Albedo @ SSIM: 0.9332
Armadillo Albedo @ LPIPS: 0.1004
Ficus RLIT[1] @ PSNR: 21.173
Ficus RLIT[1] @ SSIM: 0.9168
Ficus RLIT[1] @ LPIPS: 0.0753
Ficus RLIT[2] @ PSNR: 20.620
Ficus RLIT[2] @ SSIM: 0.9051
Ficus RLIT[2] @ LPIPS: 0.0811
Ficus RLIT[3] @ PSNR: 20.676
Ficus RLIT[3] @ SSIM: 0.9173
Ficus RLIT[3] @ LPIPS: 0.0748
Ficus RLIT[4] @ PSNR: 20.945
Ficus RLIT[4] @ SSIM: 0.9133
Ficus RLIT[4] @ LPIPS: 0.0749
Ficus RLIT[5] @ PSNR: 20.698
Ficus RLIT[5] @ SSIM: 0.9211
Ficus RLIT[5] @ LPIPS: 0.0725
Ficus Albedo @ PSNR: 25.614
Ficus Albedo @ SSIM: 0.9148
Ficus Albedo @ LPIPS: 0.0873
Hotdog RLIT[1] @ PSNR: 28.681
Hotdog RLIT[1] @ SSIM: 0.9388
Hotdog RLIT[1] @ LPIPS: 0.0725
Hotdog RLIT[2] @ PSNR: 28.202
Hotdog RLIT[2] @ SSIM: 0.9474
Hotdog RLIT[2] @ LPIPS: 0.0763
Hotdog RLIT[3] @ PSNR: 31.332
Hotdog RLIT[3] @ SSIM: 0.9179
Hotdog RLIT[3] @ LPIPS: 0.0641
Hotdog RLIT[4] @ PSNR: 31.525
Hotdog RLIT[4] @ SSIM: 0.9448
Hotdog RLIT[4] @ LPIPS: 0.0784
Hotdog RLIT[5] @ PSNR: 33.613
Hotdog RLIT[5] @ SSIM: 0.9641
Hotdog RLIT[5] @ LPIPS: 0.0512
Hotdog Albedo @ PSNR: 28.412
Hotdog Albedo @ SSIM: 0.9618
Hotdog Albedo @ LPIPS: 0.0554
Lego RLIT[1] @ PSNR: 25.979
Lego RLIT[1] @ SSIM: 0.8912
Lego RLIT[1] @ LPIPS: 0.0857
Lego RLIT[2] @ PSNR: 22.690
Lego RLIT[2] @ SSIM: 0.8915
Lego RLIT[2] @ LPIPS: 0.0936
Lego RLIT[3] @ PSNR: 28.198
Lego RLIT[3] @ SSIM: 0.8702
Lego RLIT[3] @ LPIPS: 0.0636
Lego RLIT[4] @ PSNR: 27.848
Lego RLIT[4] @ SSIM: 0.9174
Lego RLIT[4] @ LPIPS: 0.0650
Lego RLIT[5] @ PSNR: 28.382
Lego RLIT[5] @ SSIM: 0.9221
Lego RLIT[5] @ LPIPS: 0.0487
Lego Albedo @ PSNR: 24.542
Lego Albedo @ SSIM: 0.9095
Lego Albedo @ LPIPS: 0.1015
'''

if __name__ == '__main__':

    def avg(l, fixed):
        fmt = f'.{fixed}f'
        return f'{sum(l) / len(l):{fmt}}'

    def display(l):
        print(' & '.join([str(x) for x in l]))

    lut = {}
    for key, value in [line.split(': ') for line in text_tsir.split('\n') if line]:
        lut.setdefault(key.split(' ', 1)[0], {})[key.split(' ', 1)[1]] = float(value)

    scenes = ['Lego', 'Hotdog', 'Armadillo', 'Ficus']

    print('Relight')
    lst = []
    for scene in scenes:
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ PSNR'] for i in range(5)]) / 5, 3))
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ SSIM'] for i in range(5)]) / 5, 4))
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ LPIPS'] for i in range(5)]) / 5, 4))
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)

    print('Albedo')
    lst = []
    for scene in scenes:
        lst.append(lut[scene]['Albedo @ PSNR'])
        lst.append(lut[scene]['Albedo @ SSIM'])
        lst.append(lut[scene]['Albedo @ LPIPS'])
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)