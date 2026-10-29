import cv2
import torch
import numpy as np
from ultralytics import YOLO

# YOLO modelini yukle
model = YOLO("yolo11n.pt")

# MiDaS icin cihaz ve model ayarla
if torch.cuda.is_available():
    cihaz = torch.device("cuda")
else:
    cihaz = torch.device("cpu")

print("MiDaS modeli yukleniyor...")

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.to(cihaz)
midas.eval()

donusumler = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

donusum = donusumler.small_transform

print("MiDaS hazir!")

# Kamerayi ac
kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    print("Kamera acilamadi.")
    raise SystemExit

while True:
    basarili, goruntu = kamera.read()

    if not basarili:
        print("Kamera goruntusu alinamadi.")
        break

    # Ayna goruntusu
    goruntu = cv2.flip(goruntu, 1)

    goruntu_yuksekligi, goruntu_genisligi = goruntu.shape[:2]

    # MiDaS ile derinlik tahmini yap
    rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    girdi = donusum(rgb).to(cihaz)

    with torch.no_grad():
        tahmin = midas(girdi)

        tahmin = torch.nn.functional.interpolate(
            tahmin.unsqueeze(1),
            size=(goruntu_yuksekligi, goruntu_genisligi),
            mode="bicubic",
            align_corners=False
        ).squeeze()

    derinlik = tahmin.cpu().numpy()

    # Derinlik degerlerini 0-1 araligina getir
    alt_sinir = np.percentile(derinlik, 5)
    ust_sinir = np.percentile(derinlik, 95)

    if ust_sinir > alt_sinir:
        derinlik_norm = np.clip(
            (derinlik - alt_sinir) / (ust_sinir - alt_sinir),
            0,
            1
        )
    else:
        derinlik_norm = np.zeros_like(derinlik)

    # Derinlik haritasini renkli goster
    derinlik_goruntu = (
        derinlik_norm * 255
    ).astype(np.uint8)

    renkli_derinlik = cv2.applyColorMap(
        derinlik_goruntu,
        cv2.COLORMAP_MAGMA
    )

    # YOLO ile nesneleri bul
    sonuclar = model(
        goruntu,
        imgsz=640,
        verbose=False
    )

    sonuc = sonuclar[0]

    for kutu in sonuc.boxes:

        guven = float(kutu.conf[0])

        if guven < 0.30:
            continue

        sinif_no = int(kutu.cls[0])
        nesne_adi = sonuc.names[sinif_no]

        x1, y1, x2, y2 = map(
            int,
            kutu.xyxy[0].tolist()
        )

        # Koordinatlari goruntu sinirlari icinde tut
        x1 = max(0, min(x1, goruntu_genisligi - 1))
        x2 = max(0, min(x2, goruntu_genisligi))
        y1 = max(0, min(y1, goruntu_yuksekligi - 1))
        y2 = max(0, min(y2, goruntu_yuksekligi))

        # Nesnenin yatay merkezini hesapla
        merkez_x = (x1 + x2) // 2

        if merkez_x < goruntu_genisligi / 3:
            konum = "sol"
        elif merkez_x < goruntu_genisligi * 2 / 3:
            konum = "orta"
        else:
            konum = "sag"

        # Nesnenin bolgesindeki derinligi bul
        nesne_derinligi = derinlik_norm[y1:y2, x1:x2]

        if nesne_derinligi.size > 0:
            derinlik_degeri = float(
                np.median(nesne_derinligi)
            )

            if derinlik_degeri >= 0.65:
                mesafe = "yakin"
            elif derinlik_degeri >= 0.35:
                mesafe = "orta"
            else:
                mesafe = "uzak"
        else:
            mesafe = "bilinmiyor"

        # Ekranda gosterilecek yazi
        yazi = f"{nesne_adi} - {konum} - {mesafe} - %{guven * 100:.0f}"

        # Nesnenin etrafina kutu ciz
        cv2.rectangle(
            goruntu,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Nesne bilgisini ekrana yaz
        cv2.putText(
            goruntu,
            yazi,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Kamera ve derinlik haritasini goster
    cv2.imshow("Gorsel Asistan", goruntu)
    cv2.imshow("MiDaS Derinlik Haritasi", renkli_derinlik)

    # Q tusuyla cik
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamera ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()