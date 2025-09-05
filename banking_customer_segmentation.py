"""
Banking Customer Segmentation Analysis
=====================================

This script performs comprehensive customer segmentation for banking customers
using the provided banking dataset with 24 features including demographics,
financial behavior, and engagement metrics.
"""

import csv
import random
import math
from collections import defaultdict, Counter

def load_banking_data(file_path):
    """Load banking customer data from CSV file"""
    customers = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numeric fields
            customer = {
                'customerid': row['customerid'],
                'age': int(row['age']),
                'gender': row['gender'],
                'income': int(row['income']),
                'occupation': row['occupation'],
                'location': row['location'],
                'householdsize': int(row['householdsize']),
                'accounttype': row['accounttype'],
                'accounttenureyears': int(row['accounttenureyears']),
                'avgbalance': float(row['avgbalance']),
                'balancevolatility': float(row['balancevolatility']),
                'loantype': row['loantype'],
                'loanamount': float(row['loanamount']),
                'creditscore': int(row['creditscore']),
                'creditutilizationratio': float(row['creditutilizationratio']),
                'delinquencycount': int(row['delinquencycount']),
                'monthlytransactions': int(row['monthlytransactions']),
                'digitalusage': int(row['digitalusage']),
                'channelpreference': row['channelpreference'],
                'dormantdays': int(row['dormantdays']),
                'revenuecontribution': float(row['revenuecontribution']),
                'crosssellindex': int(row['crosssellindex']),
                'salarypattern': row['salarypattern'],
                'spendcategory': row['spendcategory']
            }
            customers.append(customer)
    
    return customers

def preprocess_banking_data(customers):
    """Preprocess banking data for clustering"""
    # Select key numerical features for clustering
    numerical_features = [
        'age', 'income', 'accounttenureyears', 'avgbalance', 'balancevolatility',
        'loanamount', 'creditscore', 'creditutilizationratio', 'delinquencycount',
        'monthlytransactions', 'digitalusage', 'dormantdays', 'revenuecontribution',
        'crosssellindex'
    ]
    
    # Normalize numerical features
    normalized_customers = []
    for customer in customers:
        normalized_customer = customer.copy()
        
        # Normalize each numerical feature to 0-1 scale
        normalized_customer['age_norm'] = customer['age'] / 100
        normalized_customer['income_norm'] = customer['income'] / 500000  # Assuming max income around 500k
        normalized_customer['tenure_norm'] = customer['accounttenureyears'] / 30  # Assuming max tenure 30 years
        normalized_customer['balance_norm'] = customer['avgbalance'] / 1000000  # Assuming max balance 1M
        normalized_customer['volatility_norm'] = customer['balancevolatility'] / 3  # Assuming max volatility 3
        normalized_customer['loan_norm'] = customer['loanamount'] / 1000000  # Assuming max loan 1M
        normalized_customer['credit_norm'] = customer['creditscore'] / 900  # Assuming max credit score 900
        normalized_customer['utilization_norm'] = customer['creditutilizationratio'] / 1  # Already 0-1
        normalized_customer['delinquency_norm'] = customer['delinquencycount'] / 10  # Assuming max 10 delinquencies
        normalized_customer['transactions_norm'] = customer['monthlytransactions'] / 300  # Assuming max 300 transactions
        normalized_customer['digital_norm'] = customer['digitalusage'] / 100  # Already 0-100
        normalized_customer['dormant_norm'] = 1 - (customer['dormantdays'] / 365)  # Invert for recency
        normalized_customer['revenue_norm'] = customer['revenuecontribution'] / 200000  # Assuming max revenue 200k
        normalized_customer['crosssell_norm'] = customer['crosssellindex'] / 10  # Assuming max crosssell 10
        
        normalized_customers.append(normalized_customer)
    
    return normalized_customers, numerical_features

def calculate_banking_distance(customer1, customer2, features):
    """Calculate weighted Euclidean distance for banking customers"""
    # Define weights for different features based on business importance
    weights = {
        'age_norm': 0.05,
        'income_norm': 0.15,
        'tenure_norm': 0.10,
        'balance_norm': 0.15,
        'volatility_norm': 0.05,
        'loan_norm': 0.10,
        'credit_norm': 0.10,
        'utilization_norm': 0.05,
        'delinquency_norm': 0.05,
        'transactions_norm': 0.10,
        'digital_norm': 0.05,
        'dormant_norm': 0.05
    }
    
    distance = 0
    for feature in features:
        if feature in weights:
            val1 = customer1[feature]
            val2 = customer2[feature]
            weight = weights[feature]
            distance += weight * ((val1 - val2) ** 2)
    
    return math.sqrt(distance)

def banking_kmeans(customers, k=5, max_iterations=50):
    """K-means clustering for banking customers"""
    features = ['age_norm', 'income_norm', 'tenure_norm', 'balance_norm', 'volatility_norm',
                'loan_norm', 'credit_norm', 'utilization_norm', 'delinquency_norm',
                'transactions_norm', 'digital_norm', 'dormant_norm']
    
    # Smart initialization: select diverse customers as initial centroids
    centroids = []
    centroids.append(customers[0].copy())
    
    for _ in range(k-1):
        max_distance = 0
        best_customer = None
        
        for customer in customers:
            min_distance_to_centroids = min([
                calculate_banking_distance(customer, centroid, features) 
                for centroid in centroids
            ])
            
            if min_distance_to_centroids > max_distance:
                max_distance = min_distance_to_centroids
                best_customer = customer
        
        if best_customer:
            centroids.append(best_customer.copy())
    
    # K-means iterations
    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        
        # Assign customers to nearest centroid
        for customer in customers:
            distances = [
                calculate_banking_distance(customer, centroid, features) 
                for centroid in centroids
            ]
            closest_cluster = distances.index(min(distances))
            clusters[closest_cluster].append(customer)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = {}
                for feature in features:
                    values = [customer[feature] for customer in cluster]
                    new_centroid[feature] = sum(values) / len(values)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[len(new_centroids)])
        
        # Check convergence
        converged = True
        for i in range(k):
            for feature in features:
                if abs(centroids[i][feature] - new_centroids[i][feature]) > 0.01:
                    converged = False
                    break
            if not converged:
                break
        
        centroids = new_centroids
        if converged:
            break
    
    # Assign final cluster labels
    for i, cluster in enumerate(clusters):
        for customer in cluster:
            customer['cluster'] = i
    
    return customers, centroids, clusters

def analyze_banking_clusters(customers):
    """Comprehensive analysis of banking customer clusters"""
    print("=" * 80)
    print("🏦 BANKING CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 80)
    
    # Group by clusters
    clusters = defaultdict(list)
    for customer in customers:
        clusters[customer['cluster']].append(customer)
    
    # Overall statistics
    total_customers = len(customers)
    total_revenue = sum(customer['revenuecontribution'] for customer in customers)
    total_balance = sum(customer['avgbalance'] for customer in customers)
    
    print(f"\n📊 BANKING OVERVIEW:")
    print(f"   Total Customers: {total_customers:,}")
    print(f"   Total Revenue: ₹{total_revenue:,.2f}")
    print(f"   Total Deposits: ₹{total_balance:,.2f}")
    print(f"   Average Customer Value: ₹{total_revenue/total_customers:,.2f}")
    
    # Cluster analysis
    print(f"\n🔍 CLUSTER ANALYSIS:")
    print("-" * 80)
    
    cluster_stats = []
    
    for cluster_id in sorted(clusters.keys()):
        cluster_customers = clusters[cluster_id]
        cluster_size = len(cluster_customers)
        cluster_percentage = (cluster_size / total_customers) * 100
        cluster_revenue = sum(customer['revenuecontribution'] for customer in cluster_customers)
        cluster_balance = sum(customer['avgbalance'] for customer in cluster_customers)
        cluster_avg_value = cluster_revenue / cluster_size
        
        print(f"\n🏷️  CLUSTER {cluster_id} ({cluster_size} customers - {cluster_percentage:.1f}%)")
        print(f"   Revenue Contribution: ₹{cluster_revenue:,.2f} ({cluster_revenue/total_revenue*100:.1f}%)")
        print(f"   Deposit Contribution: ₹{cluster_balance:,.2f} ({cluster_balance/total_balance*100:.1f}%)")
        print(f"   Average Customer Value: ₹{cluster_avg_value:,.2f}")
        
        # Calculate averages for key metrics
        avg_age = sum(c['age'] for c in cluster_customers) / cluster_size
        avg_income = sum(c['income'] for c in cluster_customers) / cluster_size
        avg_balance = sum(c['avgbalance'] for c in cluster_customers) / cluster_size
        avg_credit_score = sum(c['creditscore'] for c in cluster_customers) / cluster_size
        avg_tenure = sum(c['accounttenureyears'] for c in cluster_customers) / cluster_size
        avg_transactions = sum(c['monthlytransactions'] for c in cluster_customers) / cluster_size
        avg_digital_usage = sum(c['digitalusage'] for c in cluster_customers) / cluster_size
        avg_dormant_days = sum(c['dormantdays'] for c in cluster_customers) / cluster_size
        avg_crosssell = sum(c['crosssellindex'] for c in cluster_customers) / cluster_size
        avg_loan_amount = sum(c['loanamount'] for c in cluster_customers) / cluster_size
        
        print(f"   📈 Key Metrics:")
        print(f"      • Average Age: {avg_age:.1f} years")
        print(f"      • Average Income: ₹{avg_income:,.0f}")
        print(f"      • Average Balance: ₹{avg_balance:,.0f}")
        print(f"      • Average Credit Score: {avg_credit_score:.0f}")
        print(f"      • Average Tenure: {avg_tenure:.1f} years")
        print(f"      • Monthly Transactions: {avg_transactions:.1f}")
        print(f"      • Digital Usage: {avg_digital_usage:.1f}%")
        print(f"      • Dormant Days: {avg_dormant_days:.1f}")
        print(f"      • Cross-sell Index: {avg_crosssell:.1f}")
        
        # Categorical analysis
        gender_dist = Counter(c['gender'] for c in cluster_customers)
        occupation_dist = Counter(c['occupation'] for c in cluster_customers)
        location_dist = Counter(c['location'] for c in cluster_customers)
        account_type_dist = Counter(c['accounttype'] for c in cluster_customers)
        channel_dist = Counter(c['channelpreference'] for c in cluster_customers)
        
        print(f"   👥 Demographics:")
        print(f"      • Gender: {dict(gender_dist)}")
        print(f"      • Top Occupation: {occupation_dist.most_common(2)}")
        print(f"      • Top Location: {location_dist.most_common(2)}")
        print(f"      • Account Types: {dict(account_type_dist)}")
        print(f"      • Channel Preference: {dict(channel_dist)}")
        
        cluster_stats.append({
            'id': cluster_id,
            'size': cluster_size,
            'percentage': cluster_percentage,
            'revenue': cluster_revenue,
            'balance': cluster_balance,
            'avg_value': cluster_avg_value,
            'avg_age': avg_age,
            'avg_income': avg_income,
            'avg_balance': avg_balance,
            'avg_credit_score': avg_credit_score,
            'avg_tenure': avg_tenure,
            'avg_transactions': avg_transactions,
            'avg_digital_usage': avg_digital_usage,
            'avg_dormant_days': avg_dormant_days,
            'avg_crosssell': avg_crosssell,
            'avg_loanamount': avg_loan_amount
        })
    
    return cluster_stats, clusters

def generate_banking_insights(cluster_stats, clusters):
    """Generate banking-specific business insights and recommendations"""
    print(f"\n💡 BANKING BUSINESS INSIGHTS & STRATEGIC RECOMMENDATIONS")
    print("=" * 80)
    
    # Sort clusters by revenue contribution
    sorted_clusters = sorted(cluster_stats, key=lambda x: x['revenue'], reverse=True)
    
    for i, cluster in enumerate(sorted_clusters):
        cluster_id = cluster['id']
        cluster_customers = clusters[cluster_id]
        
        print(f"\n🎯 CLUSTER {cluster_id} - {cluster['size']} customers (₹{cluster['revenue']:,.0f} revenue)")
        print("-" * 60)
        
        # Determine banking segment type
        if (cluster['avg_income'] > 150000 and cluster['avg_balance'] > 200000 and 
            cluster['avg_credit_score'] > 700):
            segment_type = "💎 PREMIUM BANKING CLIENTS"
            characteristics = [
                "High income and deposit balances",
                "Excellent credit scores",
                "Long-term relationship potential"
            ]
            recommendations = [
                "🏆 Premium banking services and dedicated relationship manager",
                "💎 Exclusive investment and wealth management products",
                "🎫 Priority customer service and concierge banking",
                "📈 Advanced financial planning and advisory services"
            ]
        elif (cluster['avg_dormant_days'] > 180 and cluster['avg_transactions'] < 20):
            segment_type = "⚠️ AT-RISK CUSTOMERS"
            characteristics = [
                "Low transaction activity and high dormancy",
                "Potential churn risk",
                "Need re-engagement strategy"
            ]
            recommendations = [
                "📧 Proactive outreach and win-back campaigns",
                "🎁 Special offers and fee waivers",
                "📞 Personal banking consultation",
                "🔄 Product recommendations based on profile"
            ]
        elif (cluster['avg_digital_usage'] > 70 and cluster['avg_age'] < 40):
            segment_type = "📱 DIGITAL-FIRST CUSTOMERS"
            characteristics = [
                "High digital engagement and younger demographic",
                "Tech-savvy and mobile-first",
                "Prefer self-service channels"
            ]
            recommendations = [
                "📱 Enhanced mobile banking features and app optimization",
                "🤖 AI-powered chatbots and digital assistants",
                "💳 Digital payment solutions and fintech partnerships",
                "🎮 Gamification and rewards programs"
            ]
        elif (cluster['avg_loanamount'] > 300000 and cluster['avg_crosssell'] < 2):
            segment_type = "🏠 LOAN CUSTOMERS - CROSS-SELL OPPORTUNITY"
            characteristics = [
                "Active loan customers with low cross-sell",
                "High loan amounts but limited product usage",
                "Cross-selling potential"
            ]
            recommendations = [
                "💳 Credit card and personal loan offers",
                "🏦 Investment and insurance product recommendations",
                "📊 Financial health check and product bundling",
                "🎯 Targeted cross-sell campaigns"
            ]
        elif (cluster['avg_tenure'] > 10 and cluster['avg_balance'] > 100000):
            segment_type = "🛡️ LOYAL CUSTOMERS"
            characteristics = [
                "Long-term customers with stable balances",
                "High loyalty and low churn risk",
                "Relationship-focused banking"
            ]
            recommendations = [
                "🎁 Loyalty rewards and relationship benefits",
                "📈 Upselling to premium products and services",
                "💬 Regular relationship reviews and feedback",
                "🏆 Recognition programs and exclusive events"
            ]
        else:
            segment_type = "🛒 STANDARD BANKING CUSTOMERS"
            characteristics = [
                "Average engagement across all metrics",
                "Price-sensitive and value-conscious",
                "Regular banking needs"
            ]
            recommendations = [
                "💳 Competitive rates and fee structures",
                "📦 Bundle products and service packages",
                "📧 Educational content and financial literacy",
                "🎯 Targeted promotions and seasonal offers"
            ]
        
        print(f"Segment Type: {segment_type}")
        print(f"Key Characteristics:")
        for char in characteristics:
            print(f"   • {char}")
        
        print(f"Strategic Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
        
        # Calculate potential impact
        current_avg_value = cluster['avg_value']
        if "PREMIUM" in segment_type:
            potential_increase = current_avg_value * 0.20  # 20% increase potential
        elif "AT-RISK" in segment_type:
            potential_increase = current_avg_value * 0.25  # 25% retention value
        elif "DIGITAL" in segment_type:
            potential_increase = current_avg_value * 0.15  # 15% increase potential
        elif "CROSS-SELL" in segment_type:
            potential_increase = current_avg_value * 0.30  # 30% cross-sell potential
        else:
            potential_increase = current_avg_value * 0.10  # 10% increase potential
        
        print(f"💰 Revenue Potential: ₹{potential_increase * cluster['size']:,.0f} additional revenue")

def add_enhanced_segments(customers):
    """Add enhanced, human-readable segmentation columns"""
    for customer in customers:
        # 1. Customer Value Tier
        if customer['revenuecontribution'] > 100000:
            customer['value_tier'] = 'Premium'
        elif customer['revenuecontribution'] > 50000:
            customer['value_tier'] = 'Gold'
        elif customer['revenuecontribution'] > 25000:
            customer['value_tier'] = 'Silver'
        else:
            customer['value_tier'] = 'Bronze'
        
        # 2. Risk Level
        if customer['delinquencycount'] > 3 or customer['creditutilizationratio'] > 0.8:
            customer['risk_level'] = 'High Risk'
        elif customer['delinquencycount'] > 1 or customer['creditutilizationratio'] > 0.6:
            customer['risk_level'] = 'Medium Risk'
        else:
            customer['risk_level'] = 'Low Risk'
        
        # 3. Digital Adoption Level
        if customer['digitalusage'] > 70:
            customer['digital_level'] = 'Digital Native'
        elif customer['digitalusage'] > 40:
            customer['digital_level'] = 'Digital Adopter'
        else:
            customer['digital_level'] = 'Traditional'
        
        # 4. Engagement Status
        if customer['dormantdays'] > 180:
            customer['engagement_status'] = 'Dormant'
        elif customer['dormantdays'] > 90:
            customer['engagement_status'] = 'At Risk'
        elif customer['monthlytransactions'] > 100:
            customer['engagement_status'] = 'Highly Active'
        else:
            customer['engagement_status'] = 'Active'
        
        # 5. Life Stage
        if customer['age'] < 30:
            customer['life_stage'] = 'Young Professional'
        elif customer['age'] < 45:
            customer['life_stage'] = 'Established Professional'
        elif customer['age'] < 60:
            customer['life_stage'] = 'Pre-Retirement'
        else:
            customer['life_stage'] = 'Retired'
        
        # 6. Financial Health Score
        health_score = 0
        if customer['creditscore'] > 700:
            health_score += 3
        elif customer['creditscore'] > 600:
            health_score += 2
        else:
            health_score += 1
        
        if customer['creditutilizationratio'] < 0.3:
            health_score += 3
        elif customer['creditutilizationratio'] < 0.6:
            health_score += 2
        else:
            health_score += 1
        
        if customer['delinquencycount'] == 0:
            health_score += 3
        elif customer['delinquencycount'] < 2:
            health_score += 2
        else:
            health_score += 1
        
        if health_score >= 8:
            customer['financial_health'] = 'Excellent'
        elif health_score >= 6:
            customer['financial_health'] = 'Good'
        elif health_score >= 4:
            customer['financial_health'] = 'Fair'
        else:
            customer['financial_health'] = 'Poor'
        
        # 7. Product Potential
        if customer['crosssellindex'] < 1:
            customer['product_potential'] = 'High Cross-sell'
        elif customer['crosssellindex'] < 3:
            customer['product_potential'] = 'Medium Cross-sell'
        else:
            customer['product_potential'] = 'Low Cross-sell'
        
        # 8. Channel Preference Type
        if customer['channelpreference'] in ['Mobile', 'Web']:
            customer['channel_type'] = 'Digital First'
        elif customer['channelpreference'] == 'Branch':
            customer['channel_type'] = 'Relationship Banking'
        else:
            customer['channel_type'] = 'Self Service'
        
        # 9. Income Category
        if customer['income'] > 200000:
            customer['income_category'] = 'High Income'
        elif customer['income'] > 100000:
            customer['income_category'] = 'Upper Middle'
        elif customer['income'] > 50000:
            customer['income_category'] = 'Middle Income'
        else:
            customer['income_category'] = 'Lower Income'
        
        # 10. Customer Segment Name (Human Readable)
        cluster = customer['cluster']
        if cluster == 0:
            customer['segment_name'] = 'Premium Loyalists'
        elif cluster == 1:
            customer['segment_name'] = 'Standard Savers'
        elif cluster == 2:
            customer['segment_name'] = 'Digital Seniors'
        elif cluster == 3:
            customer['segment_name'] = 'Retirement Planners'
        else:
            customer['segment_name'] = 'Growth Seekers'
    
    return customers

def create_banking_summary_report(customers, cluster_stats):
    """Create comprehensive banking summary report"""
    print(f"\n📋 BANKING EXECUTIVE SUMMARY REPORT")
    print("=" * 80)
    
    total_customers = len(customers)
    total_revenue = sum(customer['revenuecontribution'] for customer in customers)
    total_balance = sum(customer['avgbalance'] for customer in customers)
    
    # Top performing clusters
    top_clusters = sorted(cluster_stats, key=lambda x: x['revenue'], reverse=True)[:3]
    
    print(f"🎯 KEY BANKING FINDINGS:")
    print(f"   • {total_customers:,} banking customers analyzed across {len(cluster_stats)} segments")
    print(f"   • Total customer revenue: ₹{total_revenue:,.2f}")
    print(f"   • Total customer deposits: ₹{total_balance:,.2f}")
    print(f"   • Top 3 clusters contribute {sum(c['revenue'] for c in top_clusters)/total_revenue*100:.1f}% of revenue")
    
    print(f"\n🏆 TOP PERFORMING BANKING SEGMENTS:")
    for i, cluster in enumerate(top_clusters, 1):
        print(f"   {i}. Cluster {cluster['id']}: {cluster['size']} customers, ₹{cluster['revenue']:,.0f} revenue")
    
    print(f"\n📈 BANKING STRATEGIC PRIORITIES:")
    print(f"   1. Focus on premium banking services for high-value segments")
    print(f"   2. Implement retention strategies for at-risk customers")
    print(f"   3. Enhance digital banking experience for tech-savvy customers")
    print(f"   4. Develop cross-selling programs for loan customers")
    print(f"   5. Strengthen relationship banking for loyal customers")
    
    # Add enhanced segments
    print(f"\n🔧 Adding enhanced segmentation columns...")
    customers = add_enhanced_segments(customers)
    
    # Save detailed results with enhanced columns
    with open('banking_customer_segments_enhanced.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['customerid', 'age', 'gender', 'income', 'occupation', 'location', 
                     'householdsize', 'accounttype', 'accounttenureyears', 'avgbalance', 
                     'balancevolatility', 'loantype', 'loanamount', 'creditscore', 
                     'creditutilizationratio', 'delinquencycount', 'monthlytransactions', 
                     'digitalusage', 'channelpreference', 'dormantdays', 'revenuecontribution', 
                     'crosssellindex', 'salarypattern', 'spendcategory', 'cluster',
                     'segment_name', 'value_tier', 'risk_level', 'digital_level', 
                     'engagement_status', 'life_stage', 'financial_health', 
                     'product_potential', 'channel_type', 'income_category']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for customer in customers:
            # Remove normalized fields for output
            output_customer = {k: v for k, v in customer.items() 
                             if not k.endswith('_norm')}
            writer.writerow(output_customer)
    
    print(f"\n💾 Enhanced results saved to: banking_customer_segments_enhanced.csv")
    
    # Print segment distribution
    print(f"\n📊 ENHANCED SEGMENT DISTRIBUTION:")
    print("-" * 50)
    
    # Value Tier Distribution
    value_tiers = Counter(c['value_tier'] for c in customers)
    print(f"💰 Value Tiers: {dict(value_tiers)}")
    
    # Risk Level Distribution
    risk_levels = Counter(c['risk_level'] for c in customers)
    print(f"⚠️ Risk Levels: {dict(risk_levels)}")
    
    # Digital Level Distribution
    digital_levels = Counter(c['digital_level'] for c in customers)
    print(f"📱 Digital Levels: {dict(digital_levels)}")
    
    # Engagement Status Distribution
    engagement_status = Counter(c['engagement_status'] for c in customers)
    print(f"🎯 Engagement Status: {dict(engagement_status)}")
    
    # Life Stage Distribution
    life_stages = Counter(c['life_stage'] for c in customers)
    print(f"👥 Life Stages: {dict(life_stages)}")
    
    # Financial Health Distribution
    financial_health = Counter(c['financial_health'] for c in customers)
    print(f"💚 Financial Health: {dict(financial_health)}")
    
    # Segment Names Distribution
    segment_names = Counter(c['segment_name'] for c in customers)
    print(f"🏷️ Segment Names: {dict(segment_names)}")

def main():
    """Main execution function for banking customer segmentation"""
    print("🏦 Starting Banking Customer Segmentation Analysis...")
    print("=" * 80)
    
    # Load banking data
    print("📊 Step 1: Loading banking customer data...")
    file_path = r"C:\Users\2025i\Downloads\banking_customer_segmentation_final.csv"
    customers = load_banking_data(file_path)
    print(f"   ✅ Loaded {len(customers)} banking customer records")
    
    # Preprocess data
    print("🔧 Step 2: Preprocessing banking data...")
    customers, features = preprocess_banking_data(customers)
    print(f"   ✅ Preprocessed {len(features)} numerical features")
    
    # Perform clustering
    print("🔍 Step 3: Performing K-means clustering...")
    customers, centroids, clusters = banking_kmeans(customers, k=5)
    print(f"   ✅ Clustering completed with 5 distinct segments")
    
    # Analyze clusters
    print("📈 Step 4: Conducting detailed cluster analysis...")
    cluster_stats, clusters_dict = analyze_banking_clusters(customers)
    
    # Generate insights
    print("💡 Step 5: Generating banking business insights...")
    generate_banking_insights(cluster_stats, clusters_dict)
    
    # Summary report
    print("📋 Step 6: Creating executive summary report...")
    create_banking_summary_report(customers, cluster_stats)
    
    print(f"\n🎉 BANKING ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Analyzed {len(customers)} banking customers")
    print(f"🎯 Identified {len(cluster_stats)} distinct segments")
    print(f"💰 Total customer revenue: ₹{sum(c['revenuecontribution'] for c in customers):,.2f}")
    print(f"💾 Detailed results saved to CSV file")

if __name__ == "__main__":
    main()
