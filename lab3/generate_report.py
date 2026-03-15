"""Generate Lab 3 PDF report using fpdf2."""
import json
from fpdf import FPDF

# Load results
with open('results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

class LabReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('DejaVu', 'I', 8)
            self.cell(0, 10, 'Лабораторна робота №3', 0, 0, 'C')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Сторінка {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('DejaVu', 'B', 13)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(1)

    def subsection_title(self, title):
        self.set_font('DejaVu', 'B', 11)
        self.cell(0, 7, title, 0, 1, 'L')

    def body_text(self, text):
        self.set_font('DejaVu', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font('DejaVuMono', '', 8)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(1)

    def add_plot(self, img_path, w=170):
        self.image(img_path, x=20, w=w)
        self.ln(3)


pdf = LabReport()
pdf.set_auto_page_break(auto=True, margin=15)

# Add Unicode font
pdf.add_font('DejaVu', '', 'C:/Users/Pavel/Development/SignLang1/lab3/fonts/DejaVuSans.ttf', uni=True)
pdf.add_font('DejaVu', 'B', 'C:/Users/Pavel/Development/SignLang1/lab3/fonts/DejaVuSans-Bold.ttf', uni=True)
pdf.add_font('DejaVu', 'I', 'C:/Users/Pavel/Development/SignLang1/lab3/fonts/DejaVuSans-Oblique.ttf', uni=True)
pdf.add_font('DejaVuMono', '', 'C:/Users/Pavel/Development/SignLang1/lab3/fonts/DejaVuSansMono.ttf', uni=True)

# Title page
pdf.add_page()
pdf.ln(30)
pdf.set_font('DejaVu', 'B', 20)
pdf.cell(0, 15, 'Лабораторна робота №3', 0, 1, 'C')
pdf.ln(3)
pdf.set_font('DejaVu', '', 14)
pdf.cell(0, 10, 'PCA, K-means, Autoencoder,', 0, 1, 'C')
pdf.cell(0, 10, 'RandomOverSample, SMOTE, ADASYN', 0, 1, 'C')
pdf.ln(6)
pdf.set_font('DejaVu', '', 11)
pdf.cell(0, 8, 'Аналіз ефективності навчання автокодувальника', 0, 1, 'C')
pdf.cell(0, 8, 'та порівняння класифікаторів з/без аугментації даних', 0, 1, 'C')
pdf.ln(15)
pdf.set_font('DejaVu', 'B', 12)
pdf.cell(0, 8, 'Автор: Павло Шпагін', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('DejaVu', '', 10)
pdf.cell(0, 7, f'Платформа: {results["os"]}', 0, 1, 'C')
pdf.cell(0, 7, f'TensorFlow: {results["tf_version"]}', 0, 1, 'C')
pdf.cell(0, 7, f'Python: {results["python"]}', 0, 1, 'C')

# Section 1: System info
pdf.add_page()
pdf.section_title('1. Системна інформація')

pdf.body_text(
    'Для виконання лабораторної роботи використовувалася наступна конфігурація:'
)

sys_info = (
    f'Операційна система: {results["os"]}\n'
    f'TensorFlow: {results["tf_version"]}\n'
    f'Python: {results["python"]}\n'
    f'Процесор: Intel\n'
    f'Пристрої: {results["devices"]}'
)
pdf.code_block(sys_info)

pdf.body_text(
    'Комп\'ютер працює на процесорі Intel, без виділеної відеокарти '
    '(використовується лише CPU). Версія TensorFlow 2.21.0 (tensorflow-cpu).'
)

# Section 2: Data
pdf.section_title('2. Завантаження та підготовка даних')

pdf.body_text(
    f'Дані були завантажені з файлу shaped.pickle. '
    f'Розмірність вхідних даних: {results["data_shape"]} '
    f'(297 зразків, 1200 ознак кожен).'
)

# Section 3: PCA
pdf.section_title('3. Зменшення розмірності (TruncatedSVD)')

pdf.body_text(
    f'Для зменшення розмірності було використано TruncatedSVD з 10 компонентами. '
    f'Час виконання: {results["pca_time"]:.4f}с. Розмірність після перетворення: (297, 10).'
)

pdf.code_block(
    'pca = TruncatedSVD(n_components=10)\n'
    'pca.fit(ab)\n'
    'transformed_ = pca.transform(ab)'
)

# Section 4: Clustering
pdf.section_title('4. Кластеризація (MiniBatchKMeans)')

pdf.body_text(
    f'MiniBatchKMeans з 4 кластерами був застосований до зменшених даних. '
    f'Час кластеризації: {results["cluster_time"]:.4f}с.'
)

pdf.add_plot('plot_clustering.png', w=100)

pdf.body_text(
    'На рисунку видно чітке розділення даних на 4 кластери у просторі '
    'перших двох компонент TruncatedSVD.'
)

# Section 5: Autoencoder
pdf.section_title('5. Автокодувальник (Autoencoder)')

pdf.body_text(
    'Автокодувальник з архітектурою 10→60→60→2→60→60→10 був навчений '
    'протягом 500 епох з batch_size=50.'
)

pdf.code_block(
    'def create_dense_ae():\n'
    '    hidden_dim = 60\n'
    '    encoding_dim = 2\n'
    '    inp = Input(shape=(10,))\n'
    '    flat = Flatten()(inp)\n'
    '    hidden = Dense(hidden_dim, activation=\'relu\')(flat)\n'
    '    hidden2 = Dense(hidden_dim, activation=\'relu\')(hidden)\n'
    '    encoded = Dense(encoding_dim, activation=\'relu\')(hidden2)\n'
    '    ...\n'
    '    autoencoder = Model(inp, decoder(encoder(inp)))\n'
    '    return encoder, decoder, autoencoder'
)

pdf.subsection_title('Результати навчання:')

ae_info = (
    f'Загальний час навчання: {results["ae_total_time"]:.2f}с\n'
    f'Час першої епохи: {results["ae_first_epoch_ms"]:.1f}мс\n'
    f'Середній час епохи (з 2-ї): {results["ae_avg_epoch_from2_ms"]:.1f}мс\n'
    f'Середній час епохи (всіх): {results["ae_avg_epoch_all_ms"]:.1f}мс\n'
    f'Фінальний loss: {results["ae_final_loss"]:.4f}\n'
    f'Фінальний val_loss: {results["ae_final_val_loss"]:.4f}'
)
pdf.code_block(ae_info)

pdf.body_text(
    f'Середній час виконання однієї ітерації алгоритму навчання автокодувальника '
    f'(з другої епохи) становить {results["ae_avg_epoch_from2_ms"]:.1f} мілісекунд. '
    f'Перша епоха зайняла {results["ae_first_epoch_ms"]:.1f}мс через ініціалізацію. '
    f'Загальний час навчання 500 епох: {results["ae_total_time"]:.2f}с.'
)

pdf.subsection_title('Візуалізація прихованого простору')
pdf.add_plot('plot_autoencoder_latent.png', w=170)

pdf.body_text(
    'На графіках видно розподіл кластерів у 2D латентному просторі автокодувальника '
    'для тренувальної та тестової вибірок.'
)

pdf.subsection_title('Криві навчання')
pdf.add_plot('plot_ae_training.png', w=170)

pdf.body_text(
    'Криві Loss та Accuracy демонструють стабільне навчання. '
    'Loss монотонно зменшується, що свідчить про коректну конвергенцію моделі.'
)

# Section 6: Oversampling
pdf.section_title('6. Oversampling (RandomOverSampler, SMOTE, ADASYN)')

pdf.body_text(
    'Три методи аугментації даних були застосовані для балансування класів:'
)

os_info = (
    f'RandomOverSampler: час={results["ros_time"]:.4f}с\n'
    f'SMOTE: час={results["smote_time"]:.4f}с\n'
    f'ADASYN: час={results["adasyn_time"]:.4f}с'
)
pdf.code_block(os_info)

pdf.add_plot('plot_oversampling.png', w=170)

pdf.body_text(
    'RandomOverSampler дублює існуючі зразки, SMOTE генерує синтетичні зразки '
    'шляхом інтерполяції між сусідами, ADASYN генерує більше зразків для '
    'складніших випадків.'
)

# Section 7: Optional - Classifier comparison
pdf.section_title('7. Опціональне завдання: порівняння класифікаторів')

pdf.body_text(
    'Було порівняно 8 класифікаторів на даних X_embedded та y_pred '
    'з та без використання аугментації (RandomOverSampler, SMOTE, ADASYN). '
    'Для кожного класифікатора використовувалася 5-fold крос-валідація.'
)

pdf.code_block(
    'classifiers = {\n'
    '    RandomForest, LinearSVC, LogisticRegression,\n'
    '    KNeighbors, DecisionTree, AdaBoost,\n'
    '    GaussianNB, MLP\n'
    '}'
)

# Results table
clf_res = results['classifier_results']
pdf.subsection_title('Таблиця результатів (accuracy, 5-fold CV):')

pdf.set_font('DejaVuMono', '', 7)
header = f'{"Classifier":<20s} {"No Aug":>10s} {"ROS":>10s} {"SMOTE":>10s} {"ADASYN":>10s}'
pdf.cell(0, 5, header, 0, 1)
pdf.cell(0, 5, '-' * 70, 0, 1)

for clf_name in clf_res['No Augmentation']:
    no_aug = clf_res['No Augmentation'][clf_name]['mean']
    ros_v = clf_res['RandomOverSampler'][clf_name]['mean']
    smote_v = clf_res['SMOTE'][clf_name]['mean']
    adasyn_v = clf_res['ADASYN'][clf_name]['mean']
    line = f'{clf_name:<20s} {no_aug:>10.4f} {ros_v:>10.4f} {smote_v:>10.4f} {adasyn_v:>10.4f}'
    pdf.cell(0, 5, line, 0, 1)

pdf.ln(3)

pdf.add_plot('plot_classifier_comparison.png', w=170)

pdf.body_text(
    'Аналіз результатів показує:\n'
    '- LogisticRegression досягає найвищої точності (~99.3-99.4%) незалежно від аугментації\n'
    '- RandomForest, LinearSVC, DecisionTree та MLP показують стабільно високі результати (>97%)\n'
    '- KNeighbors показує дещо нижчу точність (~93-94%)\n'
    '- AdaBoost демонструє аномально низьку точність (~50-56%), що свідчить про '
    'непридатність цього методу для даної задачі у поточній конфігурації\n'
    '- Аугментація даних (SMOTE, ADASYN, RandomOverSampler) незначно покращує '
    'результати більшості класифікаторів'
)

# Conclusions
pdf.section_title('8. Висновки')

pdf.body_text(
    'В даній лабораторній роботі був запущений зразок проекту, що реалізує навчання '
    'декількох алгоритмів K-means, Autoencoder, RandomOverSample, SMOTE, ADASYN '
    'на вибірці зразків даних розмірністю (297, 1200).'
)

pdf.body_text(
    'Для запуску додатку використовувався Jupyter Notebook в ОС Windows 10. '
    'Оскільки комп\'ютер має процесор Intel (без виділеної відеокарти), '
    'навчання автокодувальника виконувалось на CPU з використанням tensorflow-cpu 2.21.0.'
)

pdf.body_text(
    f'Під час виконання коду було відмічено швидкодію на рівні в середньому '
    f'{results["ae_avg_epoch_from2_ms"]:.1f} мілісекунд на крок алгоритму '
    f'(з другої епохи). Перша епоха зайняла {results["ae_first_epoch_ms"]:.1f}мс '
    f'через ініціалізацію обчислювального графу TensorFlow.'
)

pdf.body_text(
    'Оскільки програмна реалізація виконалася повністю без проблем, було виконано '
    'додаткове завдання, а саме класифікація даних X_embedded, y_pred з використанням '
    '8 різних класифікаторів (RandomForest, LinearSVC, LogisticRegression, KNeighbors, '
    'DecisionTree, AdaBoost, GaussianNB, MLP) та порівняння їх ефективності з та без '
    'аугментації даних (RandomOverSampler, SMOTE, ADASYN).'
)

pdf.body_text(
    'Таким чином, можна сказати, що більшість класифікаторів (особливо LogisticRegression, '
    'RandomForest та LinearSVC) коректно класифікують надану вибірку даних на 4 класи '
    'з точністю >97%. Аугментація даних несуттєво покращує результати, оскільки базові '
    'дані вже мають достатню якість для класифікації.'
)

pdf.output('lab3_report.pdf')
print('Report generated: lab3_report.pdf')
