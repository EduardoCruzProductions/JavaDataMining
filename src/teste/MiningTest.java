package teste;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MiningTest {
	
	public static void main(String[] args) {
		
		try {
			
			//Acessando a base de dados
			DataSource dataSource = new DataSource("src/dataset/diabetes.arff");
			
			//Criando uma instância com base na base de dados
			Instances inst = dataSource.getDataSet();
			
			//Definindo o atributo classificavel
			inst.setClassIndex(8);
			
			//Definindo algoritmo de classficação
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(inst);
			
			//Criando a nova instância para teste
			Instance teste = new DenseInstance(9);
			teste.setDataset(inst);
			
			teste.setValue(0, 0); // Número de vezes grávida
			teste.setValue(1, 162); // Concentração plasmática de glicose 2 horas em teste oral de tolerância à glicose
			teste.setValue(2, 52); // Pressão arterial diastólica (mm Hg)
			teste.setValue(3, 38); // Espessura da dobra da pele do tríceps (milímetro)
			teste.setValue(4, 0); // Insulina sérica de 2 horas (mu U / ml)
			teste.setValue(5, 37.2); // Índice de massa corporal (peso em kg / (altura em m) ^ 2)
			teste.setValue(6, 0.652); // Função de pedigree de diabetes
			teste.setValue(7, 24); // Idade (Anos)
			
			//Gerando a probabilidade a partir do algoritmo de classificação
			double[] probabilidade = nb.distributionForInstance(teste);
			
			System.out.println("Probabilidades: ");
			System.out.println("Positivo - "+probabilidade[1]+"%");
			System.out.println("Negativo - "+probabilidade[0]+"%");
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
	}
	
	// Referência
	// https://iaexpert.com.br/index.php/tag/mineracao-de-dados/page/2/
	
}
