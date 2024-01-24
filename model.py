import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from PIL import Image

# Define the modified prewhiten function
def prewhiten(x):
    mean = x.mean().item()
    std = x.std().item()
    std_adj = np.clip(std, a_min=1.0/(float(x.numel())**0.5), a_max=None)
    y = (x - mean) / std_adj
    return y

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cpu')

# Initialize Inception Resnet V1 for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load images of known people
image_of_person1 = Image.open("D:\\Ajce\\project\\gitmodel\\attendencesystem\\images\\aswin.jpg")  # Replace with the actual path
image_of_person2 = Image.open("D:\\Ajce\\project\\gitmodel\\attendencesystem\\images\\jithin.jpg")  # Replace with the actual path
image_of_person3 = Image.open("D:\\Ajce\\project\\gitmodel\\attendencesystem\\images\\jeevan.jpg")
image_of_person4 = Image.open("D:\\Ajce\\project\\gitmodel\\attendencesystem\\images\\jis.jpg")

# Preprocess images
person1_faces = mtcnn(image_of_person1)
person2_faces = mtcnn(image_of_person2)
person3_faces = mtcnn(image_of_person3)
person4_faces = mtcnn(image_of_person4)

# Encode faces
if person1_faces is not None and len(person1_faces) > 0:
    person1_face = prewhiten(person1_faces[0])
    with torch.no_grad():
        person1_embedding = model(person1_face.unsqueeze(0))[0]
else:
    print("No face detected in image_of_person1")

if person2_faces is not None and len(person2_faces) > 0:
    person2_face = prewhiten(person2_faces[0])
    with torch.no_grad():
        person2_embedding = model(person2_face.unsqueeze(0))[0]
else:
    print("No face detected in image_of_person2")

if person3_faces is not None and len(person3_faces) > 0:
    person3_face = prewhiten(person3_faces[0])
    with torch.no_grad():
        person3_embedding = model(person3_face.unsqueeze(0))[0]
else:
    print("No face detected in image_of_person3")

if person4_faces is not None and len(person4_faces) > 0:
    person4_face = prewhiten(person4_faces[0])
    with torch.no_grad():
        person4_embedding = model(person4_face.unsqueeze(0))[0]
else:
    print("No face detected in image_of_person4")

# Rest of the code remains unchanged

known_face_embeddings = [person1_embedding, person2_embedding, person3_embedding, person4_embedding]
known_face_names = ["aswin.jpg", "jithin.jpg", "jeevan.jpg", "jis.jpg"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Counter to track the number of detected and encoded faces
detected_faces_count = 0

while detected_faces_count < 2:
    ret, frame = cap.read()

    # Perform face detection
    boxes, probs = mtcnn.detect(frame)

    if boxes is not None:
        print("Face detected!")

        for i, bounding_box in enumerate(boxes):
            # Extract the face bounding box
            left, top, right, bottom = map(int, bounding_box)

            # Extract the face region
            face_region = extract_face(frame, bounding_box)  # No need to cast to int

            # Preprocess the face
            face_region = prewhiten(face_region)

            # Encode the face
            with torch.no_grad():
                face_embedding = model(face_region.unsqueeze(0))[0]

            # Compare with known faces
            distances = [torch.nn.functional.pairwise_distance(face_embedding, known_embedding) for known_embedding in known_face_embeddings]
            min_distance = min(distances)

            # Display the result
            if min_distance < 1.0:  # Adjust this threshold as needed
                min_index = distances.index(min_distance)
                file_name = known_face_names[min_index]
                print(f"Detected {file_name} with distance: {min_distance.item()}")
                detected_faces_count += 1
            else:
                print("Unknown face detected")

            # Draw rectangle and label for detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            if min_distance < 1.0:
                cv2.putText(frame, file_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    else:
        print("No face detected in video")

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
