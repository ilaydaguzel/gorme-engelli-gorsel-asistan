import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("yolo11n.pt")

# Kamerayı aç
kamera = cv2.VideoCapture(0)

while True:
    basarili, goruntu = kamera.read()

    if not basarili:
        print("Kamera görüntüsü alınamadı.")
        break

    # Görüntünün genişliği
    goruntu_genisligi = goruntu.shape[1]

    # YOLO ile görüntüyü analiz et
    sonuclar = model(goruntu, imgsz=640, verbose=False)

    # İlk görüntünün sonucunu al
    sonuc = sonuclar[0]

    # Tespit edilen nesneleri incele
    for kutu in sonuc.boxes:

        # Güven değerini al
        guven = float(kutu.conf[0])

        # Çok düşük güvenilirlikli nesneleri geç
        if guven < 0.30:
            continue

        # Sınıf numarasını al
        sinif_no = int(kutu.cls[0])

        # Nesne adını al
        nesne_adi = sonuc.names[sinif_no]

        # Bounding box koordinatlarını al
        x1, y1, x2, y2 = kutu.xyxy[0]

        # Tam sayıya çevir
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Nesnenin merkez X koordinatı
        merkez_x = (x1 + x2) // 2

        if merkez_x < goruntu_genisligi / 3:
            konum = "sag"

        elif merkez_x < goruntu_genisligi * 2 / 3:
            konum = "orta"

        else:
            konum = "sol"

        # Ekranda gösterilecek yazı
        yazi = f"{nesne_adi} - {konum} - %{guven * 100:.0f}"

        # Bounding box çiz
        cv2.rectangle(
            goruntu,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Yazıyı göster
        cv2.putText(
            goruntu,
            yazi,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Kamerayı göster
    cv2.imshow("Görsel Asistan", goruntu)

    # Q tuşuyla çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamerayı kapat
kamera.release()
cv2.destroyAllWindows()

