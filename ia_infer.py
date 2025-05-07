
import os

def inferir_cbct(input_folder: str, output_folder: str):
    """
    Simula uma inferência de IA com base em arquivos CBCT no input_folder,
    e salva um achado padrão em achados.txt no output_folder.
    """

    achados_path = os.path.join(output_folder, "achados.txt")
    with open(achados_path, "w") as f:
        f.write("Achado automático: Lesão radiolúcida na região posterior da mandíbula.\n")
        f.write("Sugerida avaliação complementar com exame clínico e histórico do paciente.\n")
    
    return "Inferência concluída e achados salvos."
