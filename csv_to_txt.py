import pandas as pd
from pathlib import Path

def csv_to_txt(caminho_csv="data/car_prices.csv", caminho_txt="data/descricoes_carros.txt"):
    """
    Lê um arquivo CSV com dados de carros, converte cada linha em uma
    frase descritiva e salva todas as frases em um arquivo de texto.
    """
    # Garante que o caminho para o CSV seja um objeto Path
    arquivo_csv = Path(caminho_csv)

    # --- Etapa 1: Verificar se o arquivo CSV de entrada existe ---
    if not arquivo_csv.is_file():
        print(f"Erro: O arquivo '{caminho_csv}' não foi encontrado.")
        return 

    print(f"Lendo o arquivo '{caminho_csv}'...")
    df = pd.read_csv(arquivo_csv)

    print(f"Foram encontradas {len(df)} linhas no arquivo.")

    # --- Etapa 3: Abrir o arquivo de texto para escrita ('w') ---
    # Usar 'with open' garante que o arquivo seja fechado corretamente no final.
    # O modo 'w' (write) cria um novo arquivo ou sobrescreve um existente.
    # O encoding='utf-8' é importante para evitar problemas com acentos.
    with open(caminho_txt, 'w', encoding='utf-8') as arquivo_txt:
        
        print(f"Processando as linhas e salvando em '{caminho_txt}'...")
        
        # --- Etapa 4: Iterar por cada linha do DataFrame ---
        # O método df.iterrows() é perfeito para percorrer linha por linha
        for indice, linha in df.iterrows():
            try:
                # --- Etapa 5: Criar a frase (textinho) usando uma f-string ---
                # Usamos os valores da linha atual para preencher a frase.
                # A formatação :.2f garante que os preços fiquem com duas casas decimais.
                frase = (
                    f"On {linha['saledate']}, a {linha['year']} {linha['make']} {linha['model']} {linha['trim']} "
                    f"with a {linha['color']} exterior and {linha['interior']} interior was sold by {linha['seller']} "
                    f"in the state of {linha['state']}. This {linha['body']} model, equipped with an {linha['transmission']} transmission "
                    f"and identified by VIN {linha['vin']}, had an odometer reading of {float(linha['odometer'])} and a "
                    f"condition score of {linha['condition']}. It achieved a final selling price of ${linha['sellingprice']:.2f}, "
                    f"against a market valuation (MMR) of ${linha['mmr']:.2f}."  
                )
                
                # --- Etapa 6: Fazer o append (escrever) da frase no arquivo txt ---
                # Adicionamos '\n' no final para que cada frase fique em uma nova linha.
                arquivo_txt.write(frase + "\n")

            except KeyError as e:
                # Avisa se alguma coluna esperada não for encontrada na linha
                print(f"Aviso: Pulando a linha {indice} por falta da coluna: {e}")
            except Exception as e:
                # Captura outros possíveis erros na formatação da linha
                print(f"Aviso: Pulando a linha {indice} devido a um erro: {e}")
                
    print("\nProcesso concluído com sucesso!")


# --- Exemplo de como chamar a função ---
if __name__ == "__main__":
    # Garante que a pasta 'data' exista
    Path("data").mkdir(exist_ok=True)
    
    # Você pode chamar a função diretamente.
    # Ela usará os caminhos padrão definidos nos parâmetros.
    csv_to_txt()