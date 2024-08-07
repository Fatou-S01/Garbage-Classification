"use client";

import axios from "axios";
import { Button, FileInput, Label } from "flowbite-react";
import Image from "next/image";
import { useState } from "react";
import { MdOutlineRecycling } from "react-icons/md";

export default function Home() {

    const [image, setImage] = useState<File | null>(null);
    const [base64, setBase64] = useState("");
    const [message, setMessage] = useState("");


  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result;
        if (typeof result === "string") {
          setBase64(result.split(",")[1]);
          setImage(file);
        }
      };
      reader.readAsDataURL(file); // Lire le fichier comme une URL de données (base64)
    }
  };


  const classifyGarbagge = () => {
    if(base64){
       const garbaggeImage = {
         garbaggeImage: base64,
       };

      axios
        .post(`/api/flask`, garbaggeImage)  //METTRE LE BON URL APRES INCHALLAH
        .then((response) => {
          setMessage(`Ce dechet fait partie de la classe${response.data}`)  //METTRE LE BON response. etc...
        })
        .catch((error) => {
          setMessage("Error");
          console.error("Error fetching data:", error);
        });
    }
  }

  return (
    <main>
      <div className="bg-green-500 h-48 flex items-center justify-center">
        <MdOutlineRecycling className="text-white text-9xl mx-3" />
        <h1 className="text-6xl font-bold text-white">Garbage Classifier</h1>
      </div>
      <div className="flex items-center justify-center mt-48">
        <div className="mx-4 text-2xl font-bold">
          <Label
            htmlFor="file-upload-helper-text"
            value="Choisissez l'image d'un dechet"
          />
        </div>
        <div>
          <FileInput id="file-upload-helper-text" color="success" onChange={handleImageChange}/>
        </div>
        <div className="text-dark">
          <Button className="bg-blue-500 text-white hover:bg-blue-700 rounded-lg" onClick={classifyGarbagge}>
            <span className="mx-2">Classifier</span>
          </Button>
        </div>
      </div>
      <div className="text-2xl"><h1>{message}</h1></div>
    </main>
  );
}
