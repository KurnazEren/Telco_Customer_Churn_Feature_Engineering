# Telco_Customer_Churn_Feature_Engineering

**Business Problem**

It is desired to develop a machine learning model that can predict customers who will leave the company.
You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

**Data Set Story**

Telco customer churn data includes information about a fictitious telecom company that provided home phone and Internet services to 7043 California customers in the third quarter. It includes which customers have left, stayed or signed up for the service.
21 Variable 7043 Observation

* CustomerId : Customer ID
* Gender : Gender
* SeniorCitizen : Whether the client is old (1, 0)
* Partner : Whether the customer has a partner (Yes, No) ? whether to be married
* Dependents : Whether the customer has dependents (Yes, No) (Child, mother, father, grandmother)
* tenure : The number of months the customer has stayed with the company
* PhoneService : Whether the customer has phone service (Yes, No)
* MultipleLines : Whether the customer has more than one line (Yes, No, No Telephone service)
* InternetService: Customer's internet service provider (DSL, Fiber optic, No)
* OnlineSecurity : Whether the customer has online security (Yes, No, no Internet service)
* OnlineBackup : Whether the customer has an online backup (Yes, No, no Internet service)
* DeviceProtection : Whether the customer has device protection (Yes, No, no Internet service)
* TechSupport : Whether the customer receives technical support (Yes, No, no Internet service)
* StreamingTV : Indicates whether the customer is broadcasting TV (Yes, No, no Internet service) Indicates whether the customer uses the Internet service to stream television programs from a third-party provider
* StreamingMovies : Whether the customer is streaming movies (Yes, No, no Internet service) Indicates whether the customer is using the Internet service to stream movies from a third-party provider
* Contract : Contract duration of the client (Month to month, One year, Two years)
* PaperlessBilling : Whether the customer has a paperless invoice (Yes, No)
* PaymentMethod : Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
* MonthlyCharges : The amount charged monthly from the customer
* TotalCharges : The total amount charged from the customer
* Churn : Whether the customer uses (Yes or No) - Customers who left in the last month or quarter

**More Info:**

Each row represents a unique customer.
Variables include information about customer service, account, and demographics.
Services customers sign up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
Customer account information – how long they have been a customer, contract, payment method, paperless billing, monthly fees and total fees.
Demographic information about customers - gender, age range, and whether they have partners and dependents.

#----------------------------------------------------------------------------------------------------------------
TR:
İş Problemi

Şirketten ayrılacak müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

Veri Seti Hikayesi

Telco müşteri kaybı verileri, üçüncü çeyrekte 7043 Kaliforniya müşterisine ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkındaki bilgileri içerir. Hangi müşterilerin hizmetten ayrıldığını, kaldığını veya kaydolduğunu içerir. 21 Değişken 7043 Gözlem

CustomerId : Müşteri Kimliği
Toplumsal Cinsiyet : Cinsiyet
SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
Partner : Müşterinin partneri olup olmadığı (Evet, Hayır) ? Evli olup olmadığı
Bağımlılar : Müşterinin bakmakla yükümlü olduğu kişi olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
Görev süresi : Müşterinin şirkette kaldığı ay sayısı
PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
İnternetHizmeti: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
ÇevrimiçiGüvenlik : Müşterinin çevrimiçi güvenliğe sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineBackup : Müşterinin online yedeklemesi olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV : Müşterinin TV yayını yapıp yapmadığını belirtir (Evet, Hayır, İnternet hizmeti yok) Müşterinin İnternet hizmetini üçüncü taraf bir sağlayıcıdan televizyon programları izlemek için kullanıp kullanmadığını belirtir
StreamingMovies : Müşterinin film izleyip izlemediği (Evet, Hayır, İnternet hizmeti yok) Müşterinin İnternet hizmetini üçüncü taraf bir sağlayıcıdan film izlemek için kullanıp kullanmadığını gösterir
Sözleşme : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
TotalCharges : Müşteriden tahsil edilen toplam tutar
Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Son ay veya çeyrekte ayrılan müşteriler
Daha Fazla Bilgi:

Her satır benzersiz bir müşteriyi temsil eder. Değişkenler müşteri hizmetleri, hesap ve demografik bilgilerle ilgili bilgileri içerir. Müşterilerin kaydolduğu hizmetler - telefon, birden fazla hat, internet, çevrimiçi güvenlik, çevrimiçi yedekleme, cihaz koruması, teknik destek ve TV ve film akışı. Müşteri hesap bilgileri - ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler. Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı, ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı.
